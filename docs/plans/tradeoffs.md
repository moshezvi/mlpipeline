# Trade-offs and Limitations

## Decisions made
- Used a simple synthetic-data regression flow for reproducibility and speed.
- Used MLflow interface locally because it is lightweight for this scope.
- Kept deployment local + Docker-focused rather than full cloud provisioning.

## Trade-offs
- Fast delivery over deep model optimization.
- Conceptual CI/CD promotion notes over a full environment rollout.
- Design-level monitoring instead of a running production observability stack.

## Limitations
- No persistent model registry backend configured beyond local artifacts.
- No authentication/rate limiting on API for this exercise.
- No end-to-end integration with cloud secret management.

## Next improvements
- Add model registry stages (`staging`, `production`) in tracking backend.
- Add contract tests and stricter schema validation.
- Add real monitoring backend and alert routing.
