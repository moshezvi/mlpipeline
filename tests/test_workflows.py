from pathlib import Path


WORKFLOW_DIR = Path(".github/workflows")
WORKFLOW_FILES = (
    WORKFLOW_DIR / "train.yml",
    WORKFLOW_DIR / "inference.yml",
    WORKFLOW_DIR / "promote.yml",
)


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _run_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if not stripped.startswith("run:"):
            index += 1
            continue

        indent = _indent_width(line)
        inline = stripped.removeprefix("run:").strip()
        if inline and inline != "|":
            blocks.append(inline)
            index += 1
            continue

        block_lines = []
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if next_line.strip() and _indent_width(next_line) <= indent:
                break
            block_lines.append(next_line)
            index += 1
        blocks.append("\n".join(block_lines))
    return blocks


def test_workflow_dispatch_inputs_are_not_embedded_in_shell_scripts():
    for workflow in WORKFLOW_FILES:
        text = workflow.read_text(encoding="utf-8")
        for block in _run_blocks(text):
            assert "${{ github.event.inputs." not in block, workflow


def test_workflow_json_artifacts_are_emitted_with_json_dump():
    expected_outputs = {
        WORKFLOW_DIR / "train.yml": "training_submission.json",
        WORKFLOW_DIR / "inference.yml": "inference_build_metadata.json",
        WORKFLOW_DIR / "promote.yml": "promotion_record.json",
    }
    for workflow, output_name in expected_outputs.items():
        text = workflow.read_text(encoding="utf-8")
        assert f'open("{output_name}", "w", encoding="utf-8")' in text
        assert "json.dump(payload, f, indent=2)" in text
        assert f"cat > {output_name} <<EOF" not in text
