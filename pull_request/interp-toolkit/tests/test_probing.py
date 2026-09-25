import torch

from interp_toolkit.model import MiniTransformer, ModelConfig
from interp_toolkit.probing import (
    evaluate_probe,
    label_examples_by_keyword_pattern,
    probe_accuracy_by_layer,
    train_linear_probe,
)


def make_model(seed: int = 0) -> MiniTransformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(n_layers=3, d_model=32, n_heads=4, d_vocab=64, n_ctx=16)
    return MiniTransformer(cfg)


def test_label_examples_by_keyword_pattern():
    texts = ["a cat sat down", "a car drove by", "the cat meowed"]
    labels = label_examples_by_keyword_pattern(texts, pattern=r"\bcat\b")
    assert labels.tolist() == [1, 0, 1]


def test_train_and_evaluate_probe_runs():
    torch.manual_seed(0)
    activations = torch.randn(8, 16)
    labels = torch.randint(0, 2, (8,))

    probe = train_linear_probe(activations, labels, n_steps=20, lr=0.05)
    accuracy = evaluate_probe(probe, activations, labels)

    assert 0.0 <= accuracy <= 1.0


def test_probe_accuracy_by_layer_returns_one_entry_per_layer():
    model = make_model()
    tokens = torch.randint(0, model.cfg.d_vocab, (8, 6))
    labels = torch.randint(0, 2, (8,))

    accuracies = probe_accuracy_by_layer(model, tokens, labels, layers=[0, 1, 2])

    assert set(accuracies.keys()) == {0, 1, 2}
