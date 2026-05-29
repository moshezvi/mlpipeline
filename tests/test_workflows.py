from pathlib import Path


def _run_block_lines(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    in_block = False
    block_indent = 0

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if in_block:
            if stripped and indent <= block_indent:
                in_block = False
            else:
                yield line
                continue

        if stripped.startswith("run: |") or stripped.startswith("run: >"):
            in_block = True
            block_indent = indent
            continue

        if stripped.startswith("run:"):
            yield line


def test_workflow_dispatch_inputs_are_not_embedded_in_shell_scripts():
    workflow_dir = Path(".github/workflows")
    offenders = []

    for workflow in workflow_dir.glob("*.yml"):
        for line in _run_block_lines(workflow):
            if "github.event.inputs" in line:
                offenders.append(f"{workflow}: {line.strip()}")

    assert offenders == []
