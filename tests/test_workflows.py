from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _literal_run_blocks(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped != "run: |":
            continue

        block_lines = []
        for block_line in lines[index + 1 :]:
            if not block_line.strip():
                block_lines.append(block_line)
                continue
            block_indent = len(block_line) - len(block_line.lstrip())
            if block_indent <= indent:
                break
            block_lines.append(block_line)
        blocks.append("\n".join(block_lines))
    return blocks


def test_workflow_dispatch_inputs_are_not_spliced_into_run_scripts():
    for path in WORKFLOW_DIR.glob("*.yml"):
        for block in _literal_run_blocks(path):
            assert "github.event.inputs." not in block, path


def test_workflow_metadata_is_not_written_with_expanding_heredocs():
    for path in WORKFLOW_DIR.glob("*.yml"):
        for block in _literal_run_blocks(path):
            assert not ("cat >" in block and "<<EOF" in block), path
