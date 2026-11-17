from services import interpretation_store as store


def test_questions_that_differ_only_in_form_share_a_key():
    assert store.normalize("  Resting   state fMRI? ") == "resting state fmri"
    assert store._key("Resting state fMRI?") == store._key("resting state  fmri")
    assert store._key("resting state fmri") != store._key("resting state eeg")


def test_cache_is_off_by_default(monkeypatch):
    monkeypatch.delenv("INTERPRETATION_CACHE", raising=False)
    assert not store.enabled()
    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    assert store.enabled()
