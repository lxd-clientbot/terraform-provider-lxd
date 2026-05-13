#!/usr/bin/env python3
"""AI agentic loop for fixing Go build errors after an LXD client update.

This script iteratively:
1. Reads build errors from a Go build
2. Identifies affected source files
3. Sends build errors + LXD client diff + source files to an AI model via OpenRouter
4. Applies the AI's suggested file edits
5. Rebuilds and repeats until the build passes or max iterations are reached

Environment variables:
    OPENROUTER_API_KEY  - API key for OpenRouter
    OPENROUTER_MODEL    - Model identifier (e.g. anthropic/claude-sonnet-4)
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

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

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


def call_openrouter(
    model: str,
    api_key: str,
    messages: list[dict],
) -> str:
    """Call OpenRouter chat completions API and return the assistant message."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lxd-clientbot/terraform-provider-lxd",
        "X-Title": "LXD Terraform Provider - Client Update Bot",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 64000,
    }

    resp = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"]


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
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("OPENROUTER_MODEL", "")
    build_errors_file = os.environ.get("BUILD_ERRORS_FILE", "build-errors.txt")
    lxd_diff_file = os.environ.get("LXD_CLIENT_DIFF_FILE", "lxd-client-diff.txt")
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "5"))

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        return 1

    if not model:
        print("ERROR: OPENROUTER_MODEL is not set.")
        return 1

    print(f"Model: {model}")
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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        print(f"\nCalling OpenRouter ({model})...")
        try:
            response = call_openrouter(model, api_key, messages)
        except requests.RequestException as exc:
            print(f"  ✗ API call failed: {exc}")
            summary_parts.append(
                f"### Iteration {iteration}\n- API call failed: {exc}\n"
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
