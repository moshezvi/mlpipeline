from pathlib import Path


WORKFLOW_DIR = Path(".github/workflows")


def _shell_run_blocks(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        indent = len(lines[i]) - len(stripped)
        if stripped == "run: |":
            block_lines = []
            i += 1
            while i < len(lines):
                next_stripped = lines[i].lstrip()
                next_indent = len(lines[i]) - len(next_stripped)
                if next_stripped and next_indent <= indent:
                    break
                block_lines.append(lines[i])
                i += 1
            blocks.append("\n".join(block_lines))
            continue
        i += 1
    return blocks


def test_workflow_dispatch_inputs_are_not_interpolated_in_shell_blocks():
    for workflow in WORKFLOW_DIR.glob("*.yml"):
        for block in _shell_run_blocks(workflow):
            assert "github.event.inputs" not in block, workflow


def test_workflow_metadata_is_serialized_as_json():
    for workflow_name in ("train.yml", "inference.yml", "promote.yml"):
        contents = (WORKFLOW_DIR / workflow_name).read_text(encoding="utf-8")
        assert "json.dumps" in contents
        assert "<<EOF" not in contents
