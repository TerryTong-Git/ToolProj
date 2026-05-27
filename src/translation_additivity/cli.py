"""Command-line entrypoint for translation additivity experiments."""

from __future__ import annotations

import sys

from src.translation_additivity.information_additivity import main as information_additivity_main
from src.translation_additivity.native_translation_additivity import main as native_translation_additivity_main

NATIVE_TRANSLATION_COMMANDS = {"native-translation", "native-vs-translated", "translation"}
INFORMATION_COMMANDS = {"information", "information-additivity"}


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in NATIVE_TRANSLATION_COMMANDS:
        sys.argv = [sys.argv[0], *argv[1:]]
        native_translation_additivity_main()
        return 0
    if argv and argv[0] in INFORMATION_COMMANDS:
        sys.argv = [sys.argv[0], *argv[1:]]
    information_additivity_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
