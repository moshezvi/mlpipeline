from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_train_workflow_uses_immutable_local_artifact_uri():
    workflow = _workflow("train.yml")

    assert "run_manifest[\"model_path\"]" in workflow
    assert "/latest/sample_model.joblib" not in workflow
    assert "MODEL_ARTIFACT_URI=${{ github.event.inputs.output_dir }}" not in workflow


def test_workflow_metadata_is_json_serialized_without_heredocs():
    workflows = {
        "train.yml": "training_submission.json",
        "promote.yml": "promotion_record.json",
        "inference.yml": "inference_build_metadata.json",
    }

    for workflow_name, metadata_file in workflows.items():
        workflow = _workflow(workflow_name)
        assert f"cat > {metadata_file} <<EOF" not in workflow
        assert "json.dump(payload" in workflow


def test_dispatch_inputs_are_not_embedded_in_shell_commands():
    workflow = _workflow("train.yml")

    assert '--data-uri "${{ github.event.inputs.data_uri }}"' not in workflow
    assert '--output-dir "${{ github.event.inputs.output_dir }}"' not in workflow

    workflow = _workflow("inference.yml")

    assert 'TAG="${{ github.event.inputs.image_tag }}"' not in workflow
    assert 'IMAGE_URI="${{ github.event.inputs.image_repository }}:${TAG}"' not in workflow
