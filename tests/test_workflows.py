from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _run_blocks(path: Path) -> list[str]:
    blocks: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_block = False
    run_indent = 0
    current: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if in_block and stripped and indent <= run_indent:
            blocks.append("\n".join(current))
            current = []
            in_block = False

        if not in_block and stripped.startswith("run: |"):
            in_block = True
            run_indent = indent
            current = []
            continue

        if in_block:
            current.append(line)

    if in_block:
        blocks.append("\n".join(current))

    return blocks


def test_workflow_dispatch_inputs_are_not_interpolated_inside_shell_scripts():
    for workflow in WORKFLOW_DIR.glob("*.yml"):
        for block in _run_blocks(workflow):
            assert "github.event.inputs" not in block, workflow


def test_release_metadata_is_serialized_as_json_not_shell_heredocs():
    for workflow_name in ("train.yml", "inference.yml", "promote.yml"):
        workflow_text = (WORKFLOW_DIR / workflow_name).read_text(encoding="utf-8")
        assert "cat >" not in workflow_text
        assert "json.dump" in workflow_text
