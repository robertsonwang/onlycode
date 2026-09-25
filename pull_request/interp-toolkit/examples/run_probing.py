"""Demo: label a handful of texts by keyword, then check how well a
linear probe at each layer can decode that label from the model's
residual stream."""

from __future__ import annotations

import torch

from interp_toolkit.model import MiniTransformer, ModelConfig
from interp_toolkit.probing import label_examples_by_keyword_pattern, probe_accuracy_by_layer

TEXTS = [
    "the cat sat on the mat",
    "quantum entanglement is weird",
    "the dog ran in the park",
    "general relativity bends spacetime",
    "a kitten chased the yarn",
    "black holes warp light",
    "puppies love to play fetch",
    "the double slit experiment",
]


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=3, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    model = MiniTransformer(cfg)
    model.eval()

    labels = label_examples_by_keyword_pattern(TEXTS, pattern=r"\b(cat|dog|kitten|puppies)\b")
    print(f"labels: {labels.tolist()}")

    # Stand-in tokenization: this toy model has no real tokenizer, so we
    # just draw a fixed-length random token sequence per example. The
    # point is to exercise the probing pipeline, not to learn anything
    # about these specific sentences.
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.d_vocab, (len(TEXTS), 6))

    accuracies = probe_accuracy_by_layer(model, tokens, labels, layers=[0, 1, 2])
    for layer, acc in accuracies.items():
        print(f"layer {layer}: probe accuracy = {acc:.3f}")


if __name__ == "__main__":
    main()
