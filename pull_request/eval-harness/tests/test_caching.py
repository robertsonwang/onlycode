from eval_harness.caching import CachingModel
from eval_harness.model_interface import DummyModel


def test_caching_model_returns_underlying_response():
    underlying = DummyModel(responses={"caching-test-1": "4"})
    model = CachingModel(underlying)

    assert model.complete("caching-test-1") == "4"


def test_caching_model_counts_hits_and_misses():
    underlying = DummyModel(responses={"caching-test-2": "4"})
    model = CachingModel(underlying)

    model.complete("caching-test-2")
    model.complete("caching-test-2")

    assert model.misses == 1
    assert model.hits == 1


def test_caching_model_avoids_redundant_calls():
    underlying = DummyModel(responses={"caching-test-3": "4"})
    model = CachingModel(underlying)

    model.complete("caching-test-3")
    model.complete("caching-test-3")

    assert underlying.calls == ["caching-test-3"]
