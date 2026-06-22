from pathlib import Path


def test_train_workflow_serializes_dispatch_inputs_safely():
    workflow = Path(".github/workflows/train.yml").read_text(encoding="utf-8")

    unsafe_snippets = [
        'echo "Selected backend: ${{ github.event.inputs.train_backend }}"',
        '--data-uri "${{ github.event.inputs.data_uri }}"',
        'with open("${{ github.event.inputs.output_dir }}/latest/manifest.json"',
        'MODEL_ARTIFACT_URI=${{ github.event.inputs.output_dir }}/latest/sample_model.joblib',
        "cat > training_submission.json <<EOF",
        '"data_uri": "${{ github.event.inputs.data_uri }}"',
    ]
    for snippet in unsafe_snippets:
        assert snippet not in workflow

    assert 'export_env("MODEL_ARTIFACT_URI", manifest["model_path"])' in workflow
    assert "json.dumps(payload, indent=2)" in workflow
