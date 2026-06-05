from pathlib import Path


WORKFLOW_PATHS = [
    Path(".github/workflows/train.yml"),
    Path(".github/workflows/inference.yml"),
    Path(".github/workflows/promote.yml"),
]


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


def test_dispatch_inputs_are_not_embedded_in_shell_blocks():
    for workflow_path in WORKFLOW_PATHS:
        workflow_text = workflow_path.read_text(encoding="utf-8")

        for block in _run_blocks(workflow_text):
            assert "github.event.inputs" not in block, workflow_path


def test_release_metadata_is_written_with_json_serialization():
    for workflow_path in WORKFLOW_PATHS:
        workflow_text = workflow_path.read_text(encoding="utf-8")

        assert "json.dump(payload, f, indent=2)" in workflow_text, workflow_path
        assert "cat >" not in workflow_text, workflow_path


def test_train_workflow_blocks_failed_quality_release_metadata():
    workflow_text = Path(".github/workflows/train.yml").read_text(encoding="utf-8")

    assert "passed_quality_evaluation" in workflow_text
    assert "Quality evaluation failed. Blocking training release metadata." in workflow_text
