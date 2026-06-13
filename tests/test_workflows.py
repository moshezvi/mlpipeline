from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = [
    ROOT / ".github" / "workflows" / "train.yml",
    ROOT / ".github" / "workflows" / "inference.yml",
    ROOT / ".github" / "workflows" / "promote.yml",
]


def _run_blocks(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if stripped.startswith("run: |"):
            run_indent = len(line) - len(stripped)
            index += 1
            block: list[str] = []
            while index < len(lines):
                block_line = lines[index]
                block_indent = len(block_line) - len(block_line.lstrip())
                if block_line.strip() and block_indent <= run_indent:
                    break
                block.append(block_line)
                index += 1
            blocks.append("\n".join(block))
            continue
        index += 1
    return blocks


def test_workflow_dispatch_inputs_are_not_interpolated_in_shell_blocks():
    for workflow in WORKFLOWS:
        for block in _run_blocks(workflow):
            assert "github.event.inputs" not in block, workflow


def test_workflow_metadata_is_serialized_as_json():
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        assert "json.dump" in text, workflow
        assert "cat >" not in text, workflow


def test_training_submission_uses_run_scoped_artifact_uri():
    text = (ROOT / ".github" / "workflows" / "train.yml").read_text(encoding="utf-8")

    assert "latest/sample_model.joblib" not in text
    assert "Refusing mutable latest model artifact URI" in text
