from pathlib import Path


WORKFLOW_DIR = Path(".github/workflows")


def _run_blocks(workflow_text: str) -> list[str]:
    blocks = []
    lines = workflow_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("run: |"):
            base_indent = len(line) - len(line.lstrip())
            block_lines = []
            index += 1
            while index < len(lines):
                next_line = lines[index]
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_stripped and next_indent <= base_indent:
                    break
                block_lines.append(next_line)
                index += 1
            blocks.append("\n".join(block_lines))
            continue
        index += 1
    return blocks


def test_workflow_dispatch_inputs_are_not_inlined_in_shell_blocks():
    offenders = []
    for workflow in WORKFLOW_DIR.glob("*.yml"):
        for block in _run_blocks(workflow.read_text(encoding="utf-8")):
            if "github.event.inputs" in block:
                offenders.append(str(workflow))

    assert not offenders


def test_workflows_do_not_emit_json_with_unquoted_heredocs():
    offenders = []
    for workflow in WORKFLOW_DIR.glob("*.yml"):
        for block in _run_blocks(workflow.read_text(encoding="utf-8")):
            if "<<EOF" in block:
                offenders.append(str(workflow))

    assert not offenders
