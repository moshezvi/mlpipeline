from pathlib import Path


WORKFLOW_DIR = Path(".github/workflows")


def _run_block_lines(workflow_text: str):
    in_run_block = False
    run_indent = 0
    for line in workflow_text.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if in_run_block:
            if stripped and indent <= run_indent:
                in_run_block = False
            else:
                yield line
                continue
        if stripped == "run: |":
            in_run_block = True
            run_indent = indent


def test_workflow_dispatch_inputs_are_not_interpolated_inside_shell_blocks():
    for workflow_path in WORKFLOW_DIR.glob("*.yml"):
        run_script = "\n".join(_run_block_lines(workflow_path.read_text(encoding="utf-8")))
        assert "${{ github.event.inputs" not in run_script, workflow_path


def test_workflow_metadata_json_is_serialized_by_python():
    for workflow_name, artifact_name in {
        "train.yml": "training_submission.json",
        "inference.yml": "inference_build_metadata.json",
        "promote.yml": "promotion_record.json",
    }.items():
        workflow_text = (WORKFLOW_DIR / workflow_name).read_text(encoding="utf-8")
        assert artifact_name in workflow_text
        assert "json.dump" in workflow_text
        assert f"cat > {artifact_name} <<EOF" not in workflow_text
