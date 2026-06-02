from pathlib import Path


def _run_blocks(workflow_text: str) -> list[str]:
    blocks: list[str] = []
    lines = workflow_text.splitlines()
    in_block = False
    block_indent = 0
    current: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if in_block:
            if not stripped or indent > block_indent:
                current.append(line)
                continue
            blocks.append("\n".join(current))
            current = []
            in_block = False

        if stripped.startswith("run: |"):
            in_block = True
            block_indent = indent

    if in_block:
        blocks.append("\n".join(current))

    return blocks


def test_train_workflow_does_not_embed_dispatch_inputs_in_shell_blocks():
    workflow_text = Path(".github/workflows/train.yml").read_text(encoding="utf-8")

    for block in _run_blocks(workflow_text):
        assert "github.event.inputs" not in block


def test_train_workflow_enforces_manifest_quality_before_submission_metadata():
    workflow_text = Path(".github/workflows/train.yml").read_text(encoding="utf-8")

    assert "passed_quality_evaluation" in workflow_text
    assert "Quality evaluation failed. Blocking training release metadata." in workflow_text
    assert "json.dump(payload, f, indent=2)" in workflow_text
    assert "cat > training_submission.json" not in workflow_text
