from pathlib import Path


WORKFLOWS = [
    Path(".github/workflows/train.yml"),
    Path(".github/workflows/inference.yml"),
    Path(".github/workflows/promote.yml"),
]


def _run_blocks(workflow_text: str) -> list[str]:
    lines = workflow_text.splitlines()
    blocks = []
    for index, line in enumerate(lines):
        if line.strip() != "run: |":
            continue
        run_indent = len(line) - len(line.lstrip())
        block_lines = []
        for block_line in lines[index + 1 :]:
            block_indent = len(block_line) - len(block_line.lstrip())
            if block_line.strip() and block_indent <= run_indent:
                break
            block_lines.append(block_line)
        blocks.append("\n".join(block_lines))
    return blocks


def test_workflow_dispatch_inputs_are_not_interpolated_inside_shell_blocks():
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        for run_block in _run_blocks(text):
            assert "${{ github.event.inputs." not in run_block, workflow


def test_workflow_metadata_files_are_serialized_as_json():
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        assert "json.dumps(payload, indent=2)" in text, workflow
        assert "<<EOF" not in text, workflow


def test_train_workflow_blocks_failed_quality_handoff_and_uses_run_artifact():
    text = Path(".github/workflows/train.yml").read_text(encoding="utf-8")

    assert 'manifest.get("passed_quality_evaluation") is not True' in text
    assert 'manifest["model_path"]' in text
    assert "/latest/sample_model.joblib" not in text
