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

"""Fix files to have the correct Apache 2.0 license header."""

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

# Expected first line (Nex-AGI copyright)
EXPECTED_FIRST_LINE = "Copyright (c) Nex-AGI. All rights reserved."

# Expected Apache License content (after any additional copyright lines)
EXPECTED_LICENSE_CONTENT = [
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


def generate_license_header(comment_prefix: str) -> List[str]:
    """Generate the complete license header with proper comment prefix."""
    header_lines = []

    # Add first line (copyright)
    header_lines.append(f"{comment_prefix} {EXPECTED_FIRST_LINE}\n")

    # Add Apache License content
    for line in EXPECTED_LICENSE_CONTENT:
        if line:
            header_lines.append(f"{comment_prefix} {line}\n")
        else:
            header_lines.append(f"{comment_prefix}\n")

    # Add blank line after header
    header_lines.append("\n")

    return header_lines


def has_shebang(line: str) -> bool:
    """Check if a line is a shebang line."""
    return line.startswith("#!")


def find_header_end(lines: List[str], comment_prefix: str) -> int:
    """Find where the current header ends (if any).

    Returns the index of the first non-comment, non-blank line.
    If there's an existing header, this will be after it.
    Also skips leading blank lines after comments.
    """
    idx = 0

    # Skip shebang if present
    if idx < len(lines) and has_shebang(lines[idx]):
        idx += 1

    # Skip existing comment block (potential header)
    in_comment_block = False
    while idx < len(lines):
        line = lines[idx].rstrip()

        # If it's a comment line, we're in the comment block
        if line.startswith(comment_prefix):
            in_comment_block = True
            idx += 1
        # If it's a blank line and we've seen comments, keep skipping
        elif line == "" and in_comment_block:
            idx += 1
        # If it's a blank line before any comments, skip it
        elif line == "" and not in_comment_block:
            idx += 1
        else:
            # Found first non-comment, non-blank line
            break

    return idx


def fix_license_header(file_path: Path) -> bool:
    """Fix the license header in a file.

    Returns True if the file was modified, False otherwise.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    comment_prefix = get_comment_prefix(file_path)

    if not original_lines:
        # Empty file, just add the header WITHOUT trailing blank line
        # end-of-file-fixer will add exactly one newline at the end
        header_lines = generate_license_header(comment_prefix)
        # Remove the trailing blank line from header since file is empty
        if header_lines and header_lines[-1] == "\n":
            header_lines = header_lines[:-1]
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(header_lines)
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False

    # Find where the current header ends (or where it should start)
    header_end_idx = find_header_end(original_lines, comment_prefix)

    # Get the remaining content after the header
    remaining_lines = original_lines[header_end_idx:]

    # Strip leading blank lines from remaining content to avoid duplicates
    while remaining_lines and remaining_lines[0].strip() == "":
        remaining_lines = remaining_lines[1:]

    # Generate the correct header
    new_header = generate_license_header(comment_prefix)

    # If there's no remaining content (file only has header), don't add trailing blank
    # Let end-of-file-fixer handle the final newline
    if not remaining_lines:
        if new_header and new_header[-1] == "\n":
            new_header = new_header[:-1]

    # Construct the new file content
    new_lines = new_header + remaining_lines

    # Check if anything changed
    if new_lines == original_lines:
        return False

    # Write the fixed content
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """Main function to fix license headers in files.

    Returns 1 if any files were modified (pre-commit hook convention).
    """
    parser = argparse.ArgumentParser(
        description="Fix files to have the correct Apache 2.0 license header"
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Filenames to fix",
    )
    args = parser.parse_args(argv)

    if not args.filenames:
        return 0

    files_modified = False

    for filename in args.filenames:
        file_path = Path(filename)

        # Skip if file doesn't exist or is not a regular file
        if not file_path.is_file():
            continue

        # Check if file type should be processed
        if file_path.suffix not in COMMENT_STYLES:
            continue

        if fix_license_header(file_path):
            print(f"Fixed license header: {filename}")
            files_modified = True

    # Return 1 if any files were modified (pre-commit hook convention)
    return 1 if files_modified else 0


if __name__ == "__main__":
    sys.exit(main())
