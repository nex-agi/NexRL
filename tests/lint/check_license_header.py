# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that files have the correct Apache 2.0 license header."""

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

# Expected license header lines (without comment markers)
EXPECTED_HEADER = [
    "Copyright (c) Nex-AGI. All rights reserved.",
    "",
    'Licensed under the Apache License, Version 2.0 (the "License");',
    "you may not use this file except in compliance with the License.",
    "You may obtain a copy of the License at",
    "",
    "    http://www.apache.org/licenses/LICENSE-2.0",
    "",
    "Unless required by applicable law or agreed to in writing, software",
    'distributed under the License is distributed on an "AS IS" BASIS,',
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "See the License for the specific language governing permissions and",
    "limitations under the License.",
]

# Comment styles for different file types
COMMENT_STYLES = {
    ".py": "#",
    ".sh": "#",
    ".yaml": "#",
    ".yml": "#",
    ".toml": "#",
}


def get_comment_prefix(file_path: Path) -> str:
    """Get the comment prefix for a file based on its extension."""
    suffix = file_path.suffix
    return COMMENT_STYLES.get(suffix, "#")


def has_shebang(file_path: Path) -> bool:
    """Check if a file has a shebang line."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            return first_line.startswith("#!")
    except Exception:
        return False


def extract_header_lines(file_path: Path, num_lines: int = 20) -> List[str]:
    """Extract the first num_lines from a file, removing comment prefixes."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = []
            comment_prefix = get_comment_prefix(file_path)

            # Read lines from the start (no longer skip shebang)
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line)

            # Strip comment prefixes and whitespace
            cleaned_lines = []
            for line in lines:
                line = line.rstrip()
                if line.startswith(comment_prefix):
                    # Remove comment prefix and strip leading whitespace
                    cleaned = line[len(comment_prefix) :].lstrip()
                    cleaned_lines.append(cleaned)
                elif line == "":
                    cleaned_lines.append("")
                else:
                    # Non-comment line encountered
                    break

            return cleaned_lines
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def check_license_header(file_path: Path) -> bool:
    """Check if a file has the correct license header."""
    header_lines = extract_header_lines(file_path, num_lines=25)

    # Check if the expected header is present in the extracted lines
    for i, expected_line in enumerate(EXPECTED_HEADER):
        if i >= len(header_lines):
            return False

        # Allow some flexibility in whitespace
        actual = header_lines[i].strip()
        expected = expected_line.strip()

        if actual != expected:
            return False

    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Main function to check license headers in files."""
    parser = argparse.ArgumentParser(
        description="Check that files have the correct Apache 2.0 license header"
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Filenames to check",
    )
    args = parser.parse_args(argv)

    if not args.filenames:
        return 0

    return_code = 0
    for filename in args.filenames:
        file_path = Path(filename)

        # Skip if file doesn't exist or is not a regular file
        if not file_path.is_file():
            continue

        # Check if file type should be checked
        if file_path.suffix not in COMMENT_STYLES:
            continue

        # Check for shebang line
        if has_shebang(file_path):
            print(f"File has shebang line (should be removed): {filename}")
            return_code = 1

        if not check_license_header(file_path):
            print(f"Warning: Missing or incorrect license header: {filename}")
            return_code = 0

    return return_code


if __name__ == "__main__":
    sys.exit(main())
