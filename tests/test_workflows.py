from pathlib import Path


WORKFLOW_DIR = Path(".github/workflows")


def _run_blocks(workflow_text: str) -> list[str]:
    lines = workflow_text.splitlines()
    blocks: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.lstrip()
        if stripped.startswith("run: |"):
            base_indent = len(line) - len(stripped)
            block_lines: list[str] = []
            idx += 1
            while idx < len(lines):
                next_line = lines[idx]
                if next_line.strip():
                    indent = len(next_line) - len(next_line.lstrip())
                    if indent <= base_indent:
                        break
                block_lines.append(next_line)
                idx += 1
            blocks.append("\n".join(block_lines))
            continue
        idx += 1
    return blocks


def test_workflow_dispatch_inputs_are_not_interpolated_inside_shell_scripts():
    offenders = []
    for workflow_path in sorted(WORKFLOW_DIR.glob("*.yml")):
        for block in _run_blocks(workflow_path.read_text(encoding="utf-8")):
            if "${{ github.event.inputs." in block:
                offenders.append(str(workflow_path))

    assert offenders == []


def test_workflow_metadata_is_serialized_without_expanding_heredocs():
    offenders = []
    for workflow_path in sorted(WORKFLOW_DIR.glob("*.yml")):
        for block in _run_blocks(workflow_path.read_text(encoding="utf-8")):
            if "cat >" in block and "<<EOF" in block:
                offenders.append(str(workflow_path))

    assert offenders == []


def test_training_release_handoff_blocks_failed_quality_runs():
    train_workflow = (WORKFLOW_DIR / "train.yml").read_text(encoding="utf-8")

    assert "Refusing release handoff for failed quality evaluation" in train_workflow
    assert 'append_env("MODEL_ARTIFACT_URI", str(manifest["model_path"]))' in train_workflow
