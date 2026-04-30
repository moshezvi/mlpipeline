# Trade-offs and Limitations

## Decisions made
- Used a simple synthetic-data regression flow for reproducibility and speed.
- Used MLflow interface locally because it is lightweight for this scope.
- Kept deployment local + Docker-focused rather than full cloud provisioning.
- Split CI into **Validate** (PR/push quality checks) and **Release** (push-to-main publishing) paths.
- Packaged training in `Dockerfile.training` to keep runtime/dependency behavior consistent across local, CI, and managed compute.
- Kept inference deployment image model-agnostic (`Dockerfile.inference`) and supplied model references at runtime.

## Trade-offs
- Fast delivery over deep model optimization.
- Conceptual CI/CD promotion notes over a full environment rollout.
- Design-level monitoring instead of a running production observability stack.
- Validate image builds are test-only (no ECR push), which reduces CI blast radius but does not prove registry push permissions on every PR.
- Push-to-main release trigger keeps flow simple and consistent, but offers less explicit manual approval control than a gated promote job.
- Training containerization improves reproducibility and portability, but adds image build/push overhead and registry dependency.
- Model-agnostic inference images simplify reuse and rollback, but add runtime dependency on external model artifact availability.
- Promoting immutable artifact versions improves traceability and rollback confidence, but requires stronger metadata discipline and explicit provenance tracking.
- Manual approval before production promotion improves control and auditability, but adds operational latency versus fully automated promotion.

## Limitations
- No persistent model registry backend configured beyond local artifacts.
- No authentication/rate limiting on API for this exercise.
- No end-to-end integration with cloud secret management.
- AWS training/inference release flow is documented as the intended path, but parts remain design-level rather than fully provisioned end-to-end automation in this repo.

## Next improvements
- Add model registry stages (`staging`, `production`) in tracking backend.
- Add contract tests and stricter schema validation.
- Add real monitoring backend and alert routing.
- Add explicit release gates/approvals if stricter change control is required.
- Add non-production ECR push smoke verification to catch auth/policy drift before mainline release.
- Add a first-class promotion record (who approved, when, from staging candidate to production artifact URI/version).
