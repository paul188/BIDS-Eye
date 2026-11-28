"""
backend/vocabulary/tools/label_case.py
--------------------------------------
Write every label of value_mappings.yaml as it reads inside a sentence:
"major depressive disorder", "Parkinson's disease", "fMRI", "Stroop task".
Chips and summaries show the labels as written (code_names.py),
so "Unipolar Depression" next to "depression" would show there.

A word in title case ("Depression", "Task") is lowered unless it is a name
(PROPER_NOUNS). Words in capitals or mixed case (ADHD, fMRI, gradCPT) and
words with digits (N400, L1) are kept. Only the label lines change; the rest
of the file is not rewritten.

    python backend/vocabulary/tools/label_case.py            # list the changes
    python backend/vocabulary/tools/label_case.py --write    # write them
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

VALUE_MAPPINGS = Path(__file__).resolve().parents[1] / "value_mappings.yaml"
# Top-level sections whose labels are shown to users.
SECTIONS = ("diagnosis", "task", "suffix", "datatype", "handedness", "sex")

# Names: people, places, languages, titles of films and series, named tests
# and studies. Lower case; a label word matches without a possessive "'s".
PROPER_NOUNS = {
    # people and eponyms
    "alzheimer", "parkinson", "huntington", "tourette", "asperger", "crohn", "down",
    "williams", "rett", "angelman", "prader", "willi", "dravet", "rasmussen", "west",
    "lennox", "gastaut", "sturge", "weber", "wilson", "friedreich", "duchenne",
    "becker", "lewy", "pick", "creutzfeldt", "jakob", "guillain", "barré", "barre",
    "charcot", "marie", "tooth", "klinefelter", "turner", "kallmann", "cushing",
    "graves", "hashimoto", "wernicke", "korsakoff", "broca", "machado", "joseph",
    "stroop", "posner", "sternberg", "simon", "eriksen", "navon", "corsi", "rey",
    "wechsler", "raven", "cattell", "benton", "hayling", "brixton", "libet",
    "kanizsa", "mooney", "gabor", "ebbinghaus", "necker", "ponzo", "müller", "lyer",
    "pavlovian", "bayesian", "hariri", "rorschach", "kandinsky", "pinel", "ward",
    "allport", "watanabe", "markov", "fitts", "hick", "likert", "snellen", "ishihara",
    "glasgow", "beck", "hamilton", "montgomery", "asberg", "trier", "iowa",
    "wisconsin", "hanoi", "london", "cambridge", "oxford", "stanford", "boston",
    "california", "columbia", "budapest", "edzna", "montreal", "minnesota",
    # languages and peoples
    "english", "french", "german", "spanish", "italian", "dutch", "chinese",
    "cantonese", "mandarin", "japanese", "korean", "farsi", "persian", "hindi",
    "arabic", "hebrew", "russian", "polish", "finnish", "swedish", "danish",
    "norwegian", "portuguese", "greek", "turkish", "braille", "quran", "huffaz",
    # films, series, books, games, apps, named datasets and studies
    "alice", "wonderland", "raiders", "lost", "ark", "le", "la", "petit", "prince",
    "luna", "partly", "cloudy", "sherlock", "merlin", "crown", "mentalist", "sloth", "buck",
    "peaky", "inscapes", "kickstarter", "cyberball", "python", "litebook",
    "neuroepo", "neuroloki", "optimum", "brocanto", "archi", "cast", "forrest",
    "gump", "despicable", "pixar", "disney", "minecraft", "tetris", "pacman",
    # units
    "hz",
}

_LABEL_LINE = re.compile(r"^(    label: )(.*)$")
_SECTION_LINE = re.compile(r"^([a-z_]+):\s*$")
_WORD = re.compile(r"[A-Za-zÀ-ÿ']+")


def _is_title_word(word: str) -> bool:
    return len(word) > 1 and word[0].isupper() and word[1:].islower()


def _name(word: str) -> bool:
    bare = word.lower()
    bare = bare[:-2] if bare.endswith("'s") else bare
    return bare in PROPER_NOUNS


def sentence_case(label: str) -> str:
    """*label* with every title-case word lowered unless it is a name
    ("The" too when a name follows: "The Crown viewing")."""
    words = list(_WORD.finditer(label))

    def kept(index: int) -> bool:
        word = words[index].group(0)
        if not _is_title_word(word) or _name(word):
            return True
        return word == "The" and index + 1 < len(words) and _name(words[index + 1].group(0))

    out, end = [], 0
    for index, match in enumerate(words):
        out.append(label[end:match.start()])
        out.append(match.group(0) if kept(index) else match.group(0).lower())
        end = match.end()
    return "".join(out) + label[end:]


def _unquoted(raw: str) -> tuple[str, str]:
    """(text, quote) of a YAML scalar written on one line."""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1], raw[0]
    return raw, ""


def rewrite(lines: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    section = ""
    out, changes = [], []
    for line in lines:
        top = _SECTION_LINE.match(line)
        if top:
            section = top.group(1)
        match = _LABEL_LINE.match(line)
        if match and section in SECTIONS:
            text, quote = _unquoted(match.group(2))
            new = sentence_case(text)
            if new != text:
                changes.append((text, new))
                line = f"{match.group(1)}{quote}{new}{quote}"
        out.append(line)
    return out, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="write the file")
    args = parser.parse_args()
    lines = VALUE_MAPPINGS.read_text(encoding="utf-8").split("\n")
    new_lines, changes = rewrite(lines)
    for old, new in changes:
        print(f"{old!r} -> {new!r}")
    print(f"{len(changes)} labels change")
    if args.write and changes:
        VALUE_MAPPINGS.write_text("\n".join(new_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
