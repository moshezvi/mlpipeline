import re
from pathlib import Path


WORKFLOWS_DIR = Path(".github/workflows")
MANUAL_WORKFLOWS = [
    WORKFLOWS_DIR / "train.yml",
    WORKFLOWS_DIR / "inference.yml",
    WORKFLOWS_DIR / "promote.yml",
]


def _run_blocks(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"\s*run:\s*\|", line):
            continue

        run_indent = len(line) - len(line.lstrip())
        block = []
        for next_line in lines[index + 1 :]:
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_line.strip() and next_indent <= run_indent:
                break
            block.append(next_line)
        yield "\n".join(block)


def test_workflow_dispatch_inputs_are_not_embedded_in_shell_scripts():
    for workflow in MANUAL_WORKFLOWS:
        for run_block in _run_blocks(workflow):
            assert "github.event.inputs" not in run_block, workflow


def test_training_release_metadata_uses_immutable_manifest_model_path():
    train_workflow = (WORKFLOWS_DIR / "train.yml").read_text(encoding="utf-8")

    assert 'model_artifact_uri = str(manifest["model_path"])' in train_workflow
    assert "/latest/sample_model.joblib" not in train_workflow
