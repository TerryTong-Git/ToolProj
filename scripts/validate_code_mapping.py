#!/usr/bin/env python3
"""
Validate code-methodology alignment locally before push.

This script checks that:
1. All key_functions in code_mapping.json exist in their implementation files
2. Referenced methodology docs and implementation files exist
3. No orphaned mappings (files that no longer exist)

Usage:
    python scripts/validate_code_mapping.py [--verbose] [--strict]
"""

import argparse
import ast
import json
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Find project root by looking for pyproject.toml or .git."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def extract_functions_and_classes(filepath: Path) -> set[str]:
    """Extract all function and class names from a Python file."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except SyntaxError as e:
        print(f"  [WARN] Syntax error in {filepath}: {e}")
        return set()

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            # Also get methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    names.add(item.name)
    return names


def validate_mapping(mapping: dict, root: Path, verbose: bool = False) -> list[str]:
    """Validate a single mapping entry."""
    errors = []
    warnings = []

    methodology_path = root / mapping["methodology"]
    implementations = mapping.get("implementation", [])
    key_functions = mapping.get("key_functions", [])

    # Check methodology doc exists
    if not methodology_path.exists():
        errors.append(f"Methodology doc not found: {mapping['methodology']}")

    # Check implementation files and key functions
    all_found_names: set[str] = set()
    for impl_path_str in implementations:
        impl_path = root / impl_path_str
        if not impl_path.exists():
            errors.append(f"Implementation file not found: {impl_path_str}")
            continue

        if impl_path.suffix == ".py":
            found_names = extract_functions_and_classes(impl_path)
            all_found_names.update(found_names)
            if verbose:
                print(f"  Found in {impl_path_str}: {sorted(found_names)}")

    # Check key functions exist
    for func in key_functions:
        if func not in all_found_names:
            errors.append(
                f"Key function '{func}' not found in implementations "
                f"for {mapping['methodology']}"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate code-methodology alignment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )
    parser.add_argument(
        "--mapping",
        default="docs/code_mapping.json",
        help="Path to code mapping file",
    )
    args = parser.parse_args()

    root = get_project_root()
    mapping_path = root / args.mapping

    if not mapping_path.exists():
        print(f"[SKIP] No code mapping file found at {mapping_path}")
        sys.exit(0)

    print(f"Validating code-methodology alignment...")
    print(f"Project root: {root}")
    print(f"Mapping file: {mapping_path}")
    print()

    with open(mapping_path) as f:
        data = json.load(f)

    mappings = data.get("mappings", [])
    all_errors: list[str] = []

    for mapping in mappings:
        methodology = mapping.get("methodology", "unknown")
        if args.verbose:
            print(f"Checking: {methodology}")

        errors = validate_mapping(mapping, root, verbose=args.verbose)
        if errors:
            print(f"[FAIL] {methodology}")
            for error in errors:
                print(f"  - {error}")
            all_errors.extend(errors)
        elif args.verbose:
            print(f"[PASS] {methodology}")

    print()
    if all_errors:
        print(f"Validation FAILED with {len(all_errors)} error(s)")
        print()
        print("To fix these errors:")
        print("  1. Add missing functions to implementation files")
        print("  2. Or update docs/code_mapping.json to reflect current state")
        print("  3. Or create missing methodology documentation")
        sys.exit(1)
    else:
        print(f"Validation PASSED - {len(mappings)} mapping(s) verified")
        sys.exit(0)


if __name__ == "__main__":
    main()
