from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _run_block_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    run_lines: list[str] = []
    in_run_block = False
    run_indent = 0

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if in_run_block:
            if stripped and indent <= run_indent:
                in_run_block = False
            else:
                run_lines.append(line)
                continue

        if stripped == "run: |":
            in_run_block = True
            run_indent = indent

    return run_lines


def test_dispatch_inputs_are_not_interpolated_inside_shell_scripts():
    offenders = []
    for workflow in WORKFLOW_DIR.glob("*.yml"):
        for line in _run_block_lines(workflow):
            if "github.event.inputs" in line:
                offenders.append(f"{workflow.name}: {line.strip()}")

    assert offenders == []


def test_train_release_metadata_uses_immutable_run_manifest_path():
    train_workflow = (WORKFLOW_DIR / "train.yml").read_text(encoding="utf-8")

    assert 'emit_env("MODEL_ARTIFACT_URI", manifest["model_path"])' in train_workflow
    assert "/latest/sample_model.joblib" not in train_workflow
