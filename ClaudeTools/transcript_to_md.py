#!/usr/bin/env python3
"""
Convert a Claude Code session JSONL transcript to a readable Markdown file.

Usage:
    python transcript_to_md.py <path/to/session.jsonl>
    python transcript_to_md.py <path/to/session.jsonl> --out /custom/output/dir

Output:
    /Users/z2i/claude/Dylan/TranscriptsAsMD/<original_stem>_<YYYYMMDD_HHMMSS>.md
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone


DEFAULT_OUT_DIR = "/Users/z2i/claude/Dylan/TranscriptsAsMD"

ROLE_LABELS = {
    "user":      "You",
    "assistant": "Claude",
}

# Block types to include from assistant content
INCLUDE_BLOCK_TYPES = {"text"}

# Message types to skip entirely
SKIP_TYPES = {
    "file-history-snapshot",
    "tool_result",
}


def fmt_timestamp(ts_str: str) -> str:
    """ISO 8601 UTC → human-friendly local time string."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%A %d %B %Y, %I:%M:%S %p %Z")
    except Exception:
        return ts_str


def extract_text_from_user(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", "").strip())
                elif block.get("type") == "tool_result":
                    # Summarise tool results briefly
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        for b in tool_content:
                            if b.get("type") == "text":
                                snippet = b.get("text", "").strip()[:300]
                                parts.append(f"_[Tool result: {snippet}{'…' if len(b.get('text','')) > 300 else ''}]_")
                    elif isinstance(tool_content, str) and tool_content.strip():
                        snippet = tool_content.strip()[:300]
                        parts.append(f"_[Tool result: {snippet}{'…' if len(tool_content) > 300 else ''}]_")
        return "\n\n".join(p for p in parts if p)
    return ""


def extract_text_from_assistant(message: dict) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content.strip()
    parts = []
    tool_calls = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text)
        elif btype == "tool_use":
            name = block.get("name", "unknown tool")
            inp = block.get("input", {})
            # Show the most useful input field
            summary = ""
            for key in ("command", "file_path", "pattern", "query", "prompt", "skill"):
                if key in inp:
                    val = str(inp[key])
                    summary = f"`{val[:120]}{'…' if len(val) > 120 else ''}`"
                    break
            if not summary and inp:
                first_key = next(iter(inp))
                val = str(inp[first_key])
                summary = f"`{val[:120]}{'…' if len(val) > 120 else ''}`"
            tool_calls.append(f"_[Tool: **{name}**{' → ' + summary if summary else ''}]_")
    if tool_calls:
        parts.append("\n".join(tool_calls))
    return "\n\n".join(p for p in parts if p)


def parse_session(path: str) -> list[dict]:
    """Parse JSONL and return list of message dicts with role, text, timestamp."""
    messages = []
    seen_uuids = set()

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)
                continue

            msg_type = obj.get("type", "")
            if msg_type in SKIP_TYPES:
                continue

            uuid = obj.get("uuid", "")
            if uuid and uuid in seen_uuids:
                continue
            if uuid:
                seen_uuids.add(uuid)

            timestamp = obj.get("timestamp", "")

            if msg_type == "user":
                msg = obj.get("message", {})
                text = extract_text_from_user(msg)
                if text:
                    messages.append({
                        "role": "user",
                        "text": text,
                        "timestamp": timestamp,
                    })

            elif msg_type == "assistant":
                msg = obj.get("message", {})
                text = extract_text_from_assistant(msg)
                if text:
                    messages.append({
                        "role": "assistant",
                        "text": text,
                        "timestamp": timestamp,
                    })

            elif msg_type == "system":
                subtype = obj.get("subtype", "")
                if subtype == "bridge_status":
                    content = obj.get("content", "")
                    messages.append({
                        "role": "system",
                        "text": content,
                        "timestamp": timestamp,
                    })

    return messages


def render_markdown(messages: list[dict], source_path: str) -> str:
    stem = os.path.basename(source_path)
    now_str = datetime.now().strftime("%A %d %B %Y, %I:%M %p")

    lines = [
        f"# Claude Session Transcript",
        f"",
        f"**Source:** `{source_path}`  ",
        f"**Exported:** {now_str}  ",
        f"**Messages:** {len(messages)}",
        f"",
        "---",
        "",
    ]

    for i, msg in enumerate(messages):
        role = msg["role"]
        text = msg["text"]
        ts = fmt_timestamp(msg["timestamp"]) if msg["timestamp"] else ""

        if role == "system":
            lines.append(f"> **System:** {text}")
            lines.append("")
            continue

        label = ROLE_LABELS.get(role, role.capitalize())
        emoji = "🧑" if role == "user" else "🤖"

        lines.append(f"## {emoji} {label}")
        if ts:
            lines.append(f"*{ts}*")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert Claude JSONL transcript to Markdown")
    parser.add_argument("jsonl_path", help="Path to the .jsonl session file")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    path = os.path.expanduser(args.jsonl_path)
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing: {path}")
    messages = parse_session(path)
    print(f"Found {len(messages)} messages")

    md = render_markdown(messages, path)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{stem}_{timestamp_suffix}.md"
    out_path = os.path.join(args.out, out_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
