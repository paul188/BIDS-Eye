from services.results.display.dataset_cards import display_name


def test_real_name_is_kept_with_whitespace_collapsed() -> None:
    assert display_name("  Resting  state\nfMRI ", "ds000001") == "Resting state fMRI"


def test_blank_or_placeholder_name_falls_back_to_accession() -> None:
    for name in (None, "", " ", "Unnamed Dataset", "untitled"):
        assert display_name(name, "ds001551") == "ds001551"


def test_no_accession_still_gives_a_title() -> None:
    assert display_name("", None) == "Untitled dataset"
