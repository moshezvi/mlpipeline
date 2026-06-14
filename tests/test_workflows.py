from pathlib import Path


WORKFLOW_DIR = Path(".github/workflows")


def _run_blocks(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("run:"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped != "run: |":
            blocks.append(stripped.removeprefix("run:").strip())
            continue

        block_lines = []
        for child in lines[idx + 1 :]:
            child_stripped = child.strip()
            child_indent = len(child) - len(child.lstrip(" "))
            if child_stripped and child_indent <= indent:
                break
            block_lines.append(child)
        blocks.append("\n".join(block_lines))
    return blocks


def test_dispatch_inputs_are_not_embedded_in_shell_scripts():
    for workflow in ("train.yml", "inference.yml", "promote.yml"):
        path = WORKFLOW_DIR / workflow
        for block in _run_blocks(path):
            assert "${{ github.event.inputs." not in block


def test_workflow_json_artifacts_are_not_shell_heredocs():
    for workflow in ("train.yml", "inference.yml", "promote.yml"):
        text = (WORKFLOW_DIR / workflow).read_text(encoding="utf-8")
        assert "<<EOF" not in text
