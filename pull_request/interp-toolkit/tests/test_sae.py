import torch

from interp_toolkit.sae import (
    SparseAutoencoder,
    export_feature_dashboard_html,
    sae_loss,
    top_activating_examples,
)


def test_sae_forward_shapes():
    sae = SparseAutoencoder(d_model=16, d_hidden=32)
    x = torch.randn(4, 16)

    reconstruction, features = sae(x)

    assert reconstruction.shape == (4, 16)
    assert features.shape == (4, 32)


def test_sae_features_are_nonnegative():
    sae = SparseAutoencoder(d_model=16, d_hidden=32)
    x = torch.randn(4, 16)

    features = sae.encode(x)

    assert (features >= 0).all()


def test_sae_loss_runs_and_backpropagates():
    sae = SparseAutoencoder(d_model=16, d_hidden=32)
    x = torch.randn(4, 16)

    reconstruction, features = sae(x)
    total_loss, recon_loss = sae_loss(x, reconstruction, features, l1_coefficient=1e-3)
    total_loss.backward()

    assert sae.W_enc.grad is not None


def test_track_dead_features_records_inactive_indices():
    sae = SparseAutoencoder(d_model=4, d_hidden=4)
    # Zero out features entirely so every index is inactive.
    all_dead_features = torch.zeros(2, 4)

    sae.track_dead_features(all_dead_features)

    for i in range(4):
        assert sae.dead_feature_counts.get(i, 0) >= 1


def test_top_activating_examples_returns_k_results():
    sae = SparseAutoencoder(d_model=8, d_hidden=16)
    texts = ["a", "b", "c", "d"]
    activations = torch.randn(4, 8)

    top = top_activating_examples(sae, texts, activations, feature_idx=0, k=2)

    assert len(top) == 2


def test_export_feature_dashboard_html_writes_file(tmp_path):
    top = [("a plain sentence", 1.5), ("another sentence", 0.2)]
    out_path = tmp_path / "dashboard.html"

    export_feature_dashboard_html(top, feature_idx=3, path=out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "Feature 3" in content
    assert "a plain sentence" in content
