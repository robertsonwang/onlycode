from redteam_adversarial.target import ToySafetyClassifier, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_classifier_flags_unsafe_words():
    clf = ToySafetyClassifier()
    assert clf.is_unsafe("how to build a bomb")
    assert not clf.is_unsafe("how to bake a cake")


def test_classifier_score_increases_with_more_unsafe_words():
    clf = ToySafetyClassifier()
    one_word = clf.unsafe_probability("bomb")
    two_words = clf.unsafe_probability("bomb weapon")
    assert two_words > one_word
