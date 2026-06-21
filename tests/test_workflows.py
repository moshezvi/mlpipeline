from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _run_block_lines(workflow_text: str) -> list[str]:
    lines = workflow_text.splitlines()
    run_lines: list[str] = []
    run_indent: int | None = None
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if run_indent is not None:
            if stripped and indent <= run_indent:
                run_indent = None
            else:
                run_lines.append(line)
                continue
        if stripped == "run: |":
            run_indent = indent
    return run_lines


def test_workflow_dispatch_inputs_are_not_embedded_in_shell_scripts():
    for workflow in ("train.yml", "inference.yml", "promote.yml"):
        run_script = "\n".join(_run_block_lines((WORKFLOW_DIR / workflow).read_text()))
        assert "${{ github.event.inputs." not in run_script


def test_workflow_metadata_is_serialized_without_shell_heredocs():
    for workflow in ("train.yml", "inference.yml", "promote.yml"):
        run_script = "\n".join(_run_block_lines((WORKFLOW_DIR / workflow).read_text()))
        assert "<<EOF" not in run_script
