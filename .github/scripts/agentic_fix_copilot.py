#!/usr/bin/env python3
"""AI agentic loop for fixing Go build errors after an LXD client update.

This script iteratively:
1. Reads build errors from a Go build
2. Identifies affected source files
3. Sends build errors + LXD client diff + source files to GitHub Copilot CLI
4. Applies the AI's suggested file edits
5. Rebuilds and repeats until the build passes or max iterations are reached

Environment variables:
    GITHUB_TOKEN        - GitHub token for Copilot authentication (or COPILOT_GITHUB_TOKEN)
    COPILOT_MODEL       - Model identifier (e.g. claude-sonnet-4.6)
    BUILD_ERRORS_FILE   - Path to file containing initial build errors
    LXD_CLIENT_DIFF_FILE - Path to file containing the LXD client diff
    MAX_ITERATIONS      - Maximum fix attempts (default: 5)
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SYSTEM_PROMPT = """\
You are a Go developer working on the Terraform Provider for LXD \
(github.com/terraform-lxd/terraform-provider-lxd). The LXD client SDK \
(github.com/canonical/lxd) has been updated and the provider code no longer \
compiles.

Your task is to fix the Go build errors caused by API changes in the LXD \
client package. You will be given:
- The diff of what changed in the LXD client SDK
- The current build errors
- The content of each affected source file

Rules:
1. Only modify files that have build errors — do not refactor unrelated code.
2. Preserve existing logic and behavior; only adapt to the new client API.
3. If a type or function was renamed, update all references.
4. If a function signature changed, update call sites to match.
5. If a type gained or lost fields, update struct literals accordingly.
6. Do not add new features or change test logic.

Respond with ONLY a JSON array of file edits. Each element must have:
- "file": the relative file path (e.g. "internal/network/resource_network.go")
- "content": the complete updated file content

Example response format:
```json
[
  {
    "file": "internal/network/resource_network.go",
    "content": "package network\\n\\nimport (...)\\n..."
  }
]
```

If no changes are needed for a file, omit it from the array. \
Return ONLY the JSON array — no markdown fences, no explanation.\
"""


def read_file(path: str) -> str:
    """Read a file and return its content."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def write_file(path: str, content: str) -> None:
    """Write content to a file."""
    Path(path).write_text(content, encoding="utf-8")


def extract_affected_files(build_errors: str) -> list[str]:
    """Extract unique file paths from Go build error output."""
    pattern = re.compile(r"^(\./)?([^\s:]+\.go):\d+", re.MULTILINE)
    files = []
    seen = set()
    for match in pattern.finditer(build_errors):
        filepath = match.group(2)
        if filepath not in seen and Path(filepath).exists():
            seen.add(filepath)
            files.append(filepath)
    return sorted(files)


def run_build() -> tuple[int, str]:
    """Run go build and return (exit_code, output)."""
    result = subprocess.run(
        ["go", "build", "./..."],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    return result.returncode, output


def call_copilot(
    model: str,
    prompt: str,
) -> str:
    """Call GitHub Copilot CLI and return the assistant response."""
    cmd = [
        "copilot",
        "-p", prompt,
        "-s",  # silent: output only the agent response
        "--allow-tool=write,edit,create,shell",
        "--no-ask-user",
    ]

    env = os.environ.copy()
    if model:
        env["COPILOT_MODEL"] = model

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Copilot CLI failed (exit {result.returncode}): {stderr}")

    return result.stdout


def parse_file_edits(response: str) -> list[dict]:
    """Parse the AI response into a list of file edits.

    Handles responses with or without markdown code fences.
    """
    # Strip markdown code fences if present.
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        edits = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"  ✗ Failed to parse AI response as JSON: {exc}")
        print(f"  Response preview: {cleaned[:500]}")
        return []

    if not isinstance(edits, list):
        print(f"  ✗ Expected JSON array, got {type(edits).__name__}")
        return []

    valid = []
    for edit in edits:
        if isinstance(edit, dict) and "file" in edit and "content" in edit:
            valid.append(edit)
        else:
            print(f"  ✗ Skipping malformed edit entry: {edit!r:.200}")

    return valid


def apply_edits(edits: list[dict]) -> list[str]:
    """Apply file edits and return list of modified file paths."""
    modified = []
    for edit in edits:
        filepath = edit["file"]
        content = edit["content"]

        if not Path(filepath).exists():
            print(f"  ⚠ Skipping non-existent file: {filepath}")
            continue

        write_file(filepath, content)
        modified.append(filepath)
        print(f"  ✓ Updated {filepath}")

    return modified


def build_user_prompt(
    build_errors: str,
    lxd_diff: str,
    affected_files: list[str],
    iteration: int,
) -> str:
    """Construct the user prompt for the AI model."""
    parts = []

    parts.append(f"## Iteration {iteration}\n")

    parts.append("## LXD Client Diff\n")
    # Truncate very large diffs to stay within context limits.
    if len(lxd_diff) > 50000:
        parts.append(lxd_diff[:50000])
        parts.append("\n... (diff truncated) ...\n")
    else:
        parts.append(lxd_diff)

    parts.append("\n## Build Errors\n")
    parts.append(build_errors)

    parts.append("\n## Affected Source Files\n")
    for filepath in affected_files:
        content = read_file(filepath)
        parts.append(f"\n### {filepath}\n```go\n{content}\n```\n")

    parts.append(
        "\nFix the build errors above. Return ONLY a JSON array of file edits."
    )

    return "\n".join(parts)


def main() -> int:
    model = os.environ.get("COPILOT_MODEL", "")
    build_errors_file = os.environ.get("BUILD_ERRORS_FILE", "build-errors.txt")
    lxd_diff_file = os.environ.get("LXD_CLIENT_DIFF_FILE", "lxd-client-diff.txt")
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "5"))

    # Verify Copilot CLI is available.
    try:
        subprocess.run(
            ["copilot", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: GitHub Copilot CLI ('copilot') is not available.")
        print("Install it or ensure it is on PATH.")
        return 1

    print(f"Model: {model or 'default'}")
    print(f"Max iterations: {max_iterations}")

    build_errors = read_file(build_errors_file)
    lxd_diff = read_file(lxd_diff_file) if Path(lxd_diff_file).exists() else ""

    summary_parts = ["## AI Fix Summary\n"]
    all_modified: set[str] = set()
    final_success = False

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{max_iterations}")
        print(f"{'='*60}")

        affected_files = extract_affected_files(build_errors)
        if not affected_files:
            print("No affected Go files found in build errors.")
            print("Build errors may not be file-specific. Dumping errors:")
            print(build_errors)
            break

        print(f"Affected files ({len(affected_files)}):")
        for f in affected_files:
            print(f"  - {f}")

        user_prompt = build_user_prompt(
            build_errors, lxd_diff, affected_files, iteration,
        )

        print(f"\nCalling GitHub Copilot CLI ({model or 'default'})...")
        try:
            response = call_copilot(model, user_prompt)
        except RuntimeError as exc:
            print(f"  ✗ Copilot CLI call failed: {exc}")
            summary_parts.append(
                f"### Iteration {iteration}\n- Copilot CLI call failed: {exc}\n"
            )
            continue

        edits = parse_file_edits(response)
        if not edits:
            print("  ✗ No valid edits returned by AI.")
            summary_parts.append(
                f"### Iteration {iteration}\n- No valid edits returned\n"
            )
            continue

        print(f"\nApplying {len(edits)} edit(s):")
        modified = apply_edits(edits)
        all_modified.update(modified)

        summary_parts.append(
            f"### Iteration {iteration}\n- Modified: {', '.join(modified)}\n"
        )

        print("\nRebuilding...")
        exit_code, build_output = run_build()

        if exit_code == 0:
            print("✅ Build succeeded!")
            summary_parts.append("- **Result: Build passed** ✅\n")
            final_success = True
            break

        print(f"Build still failing (exit code {exit_code}).")
        build_errors = build_output

        # Update the build errors file for the workflow.
        write_file(build_errors_file, build_errors)
        summary_parts.append("- Result: Build still failing\n")

    # Write fix summary.
    summary_parts.append(f"\n**Files modified:** {len(all_modified)}\n")
    for f in sorted(all_modified):
        summary_parts.append(f"- `{f}`\n")

    write_file("fix-summary.md", "".join(summary_parts))

    if final_success:
        print("\n✅ AI fix completed successfully.")
        return 0

    print("\n⚠️  AI fix incomplete after all iterations.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
