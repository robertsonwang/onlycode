"""
WebSocket-based interactive debug runner.

Spawns the user's code in a PTY so that:
  - Full stack traces are visible
  - Interactive debuggers (pdb, breakpoint()) work
  - The user can type input in the terminal
"""

from __future__ import annotations

import asyncio
import json
import os
import pty
import select
import signal
import shutil
import struct
import sys
import fcntl
import termios
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

PROBLEMS_DIR = Path(__file__).resolve().parent / "problems"


def _build_debug_script(code: str, tests: list[dict[str, Any]]) -> str:
    """Build a script that runs user code with full tracebacks and pdb support."""
    return f'''
import json
import sys
import traceback
import time

print("\\033[1;36m=== Debug Runner ===\\033[0m")
print()

USER_CODE = {code!r}
TESTS = json.loads({json.dumps(tests)!r})

namespace = {{}}

print("\\033[33mLoading your code...\\033[0m")
try:
    exec(compile(USER_CODE, "<your_code>", "exec"), namespace)
    print("\\033[32m✓ Code loaded successfully\\033[0m")
except Exception:
    print("\\033[1;31m✗ Error loading code:\\033[0m")
    print()
    traceback.print_exc()
    sys.exit(1)

print()

solver = namespace.get("solve")
if not callable(solver):
    print("\\033[1;31m✗ Your code must define a callable solve(input_data).\\033[0m")
    sys.exit(1)

passed = 0
failed = 0

for idx, case in enumerate(TESTS, start=1):
    test_input = case.get("input")
    expected = case.get("expected_output")

    print(f"\\033[1m--- Test {{idx}} ---\\033[0m")
    print(f"  Input: {{json.dumps(test_input)}}")

    try:
        start = time.perf_counter()
        actual = solver(test_input)
        elapsed = (time.perf_counter() - start) * 1000

        if actual == expected:
            print(f"  \\033[32m✓ PASS\\033[0m  ({{elapsed:.1f}} ms)")
            passed += 1
        else:
            print(f"  \\033[31m✗ FAIL\\033[0m  ({{elapsed:.1f}} ms)")
            print(f"  Expected: {{json.dumps(expected)}}")
            print(f"  Actual:   {{repr(actual)}}")
            failed += 1
    except Exception:
        failed += 1
        print(f"  \\033[1;31m✗ EXCEPTION:\\033[0m")
        print()
        traceback.print_exc()
        print()

    print()

print("\\033[1m" + "=" * 40 + "\\033[0m")
if failed == 0:
    print(f"\\033[1;32mAll {{passed}} tests passed!\\033[0m")
else:
    print(f"\\033[32m{{passed}} passed\\033[0m, \\033[31m{{failed}} failed\\033[0m")
print()
'''


async def handle_debug_session(websocket: WebSocket) -> None:
    """Handle a WebSocket debug session."""
    await websocket.accept()

    try:
        # Wait for initial message with code and problem_id
        init_msg = await websocket.receive_text()
        payload = json.loads(init_msg)

        code = payload.get("code", "")
        problem_id = payload.get("problem_id", "")

        # Load tests
        tests_path = PROBLEMS_DIR / problem_id / "tests.json"
        if not tests_path.exists():
            await websocket.send_text("\r\n\033[31mError: tests.json not found for this problem.\033[0m\r\n")
            await websocket.close()
            return

        tests = json.loads(tests_path.read_text(encoding="utf-8"))
        script = _build_debug_script(code, tests)

        # Find python
        python_cmd = shutil.which("python3") or sys.executable

        # Spawn PTY
        master_fd, slave_fd = pty.openpty()

        # Set initial terminal size
        winsize = struct.pack("HHHH", 40, 120, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        pid = os.fork()
        if pid == 0:
            # Child process
            os.close(master_fd)
            os.setsid()

            # Set up slave as controlling terminal
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

            os.dup2(slave_fd, 0)
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            if slave_fd > 2:
                os.close(slave_fd)

            os.execvp(python_cmd, [python_cmd, "-u", "-c", script])
            os._exit(1)

        # Parent process
        os.close(slave_fd)

        # Make master_fd non-blocking
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Read from PTY and send to WebSocket
        async def read_pty():
            loop = asyncio.get_event_loop()
            try:
                while True:
                    await asyncio.sleep(0.02)
                    try:
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        await websocket.send_bytes(data)
                    except OSError:
                        # Check if child is still running
                        result = os.waitpid(pid, os.WNOHANG)
                        if result[0] != 0:
                            break
                        continue
            except (WebSocketDisconnect, Exception):
                pass

        # Write from WebSocket to PTY
        async def write_pty():
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    data = msg.get("bytes") or msg.get("text", "").encode()
                    if data:
                        os.write(master_fd, data if isinstance(data, bytes) else data.encode())
            except (WebSocketDisconnect, Exception):
                pass

        # Handle terminal resize messages
        async def handle_resize():
            # This is handled inline in write_pty via special messages
            pass

        # Run read and write concurrently
        read_task = asyncio.create_task(read_pty())
        write_task = asyncio.create_task(write_pty())

        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [read_task, write_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        # Clean up
        try:
            os.close(master_fd)
        except OSError:
            pass

        try:
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
        except (OSError, ChildProcessError):
            pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\r\n\033[31mDebug session error: {e}\033[0m\r\n")
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
