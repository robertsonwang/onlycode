# OnlyCode

Local web app for practicing medium-difficulty LeetCode-style Python problems.

## Stack

- Backend: FastAPI
- Database: SQLite
- Execution: subprocess + resource limits (`resource.setrlimit`)
- Frontend: single-page HTML + JS + Monaco Editor (CDN)

## Setup with uv

1. Create/update the environment and install dependencies:

   uv sync

2. Start backend API:

   cd backend
   uv run uvicorn main:app --reload --port 8000

3. Open frontend in browser:

   Open `frontend/index.html` directly, or run a static file server and open it.

## GPU exercises

Problems whose titles begin with `[GPU]` require an NVIDIA CUDA GPU and fail
explicitly when CUDA is unavailable. On the GPU host, install their optional
PyTorch and Hugging Face dependencies before starting the backend:

   uv sync --extra gpu

The GPU problems use a longer per-submission timeout and disable the generic
virtual-memory limit because CUDA reserves large virtual address ranges.

## API

- GET /problems
- GET /problems/{id}
- POST /submit with JSON body:

  {
    "problem_id": "two_sum_variant",
    "code": "def solve(input_data):\n    return []"
  }
