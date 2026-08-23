# Remote ML Services — Integration Status

All three remote services are live and reachable. Schemas have been
confirmed from their OpenAPI specs. Tool implementations are complete.

## Services integrated

| Service | URL | Endpoint | Method | Status |
|---|---|---|---|---|
| Risk Prediction Model | https://api.sidanex.com/cts-ml1/ | /predict | POST | Live |
| Prioritization Model  | https://api.sidanex.com/cts-ml3/ | /member/{member_id} | GET | Live |
| Star Impact Model     | https://api.sidanex.com/cts-ml4/ | /simulate/{measure_code}/scenarios | GET | Live |

## Tool files

- `tools/ganesh_tool.py`      — Risk Prediction Model (POST /predict, PatientProfile schema)
- `tools/prediction_tool.py`  — Prioritization Model  (GET /member/{id})
- `tools/star_impact_tool.py` — Star Impact Model      (GET /simulate/{measure}/scenarios)

## If a service goes down or schema changes

1. Run `python fetch_specs.py` — it will attempt `/openapi.json` and `/docs`
   for each service and save the specs to JSON files.

2. For Risk Prediction Model: check `PatientProfile` fields in
   `risk_prediction_model_openapi.json`. Update `_build_payload()` in
   `tools/ganesh_tool.py` to match any new required fields.

3. For Prioritization Model: the API has no POST endpoint — it is a
   portfolio-level read API. If a member_id is not found (404), the tool
   returns `status: unavailable` gracefully. No schema changes expected.

4. For Star Impact Model: check `SimulationRequest` and
   `DirectionalStarImpactOutput` in `star_impact_model_openapi.json`.
   Update `GAP_TO_MEASURE` mapping in `tools/star_impact_tool.py` if
   new CMS measure codes are added.

5. Update `REMOTE_SERVICES` in `agent/health_check.py` if URLs change.

## No other files need to change

The orchestrator, context builder, reasoning prompt, and FastAPI routes
are all generic — they use the tool registry and treat all models uniformly
by their `status` field. Adding or removing a tool only requires editing
`agent/orchestrator.py` (imports + TOOL_REGISTRY) and `agent/context_builder.py`
(function signature + tool_outputs dict).
