# AI Hub Integration Notes

## Required Inputs

- `aihub.base_url`
- `aihub.project`
- API key in env var (default: `QAI_HUB_API_KEY`)

## Config Switch

Set in YAML:

```yaml
aihub:
  mode: qai_hub
  base_url: https://<your-ai-hub-endpoint>
  project: <project-name>
```

Set env:

```bash
export QAI_HUB_API_KEY=<token>
```

## Endpoint Contract Alignment

`QAIHubProfilerBackend` assumes these endpoints:
- `POST /v1/artifacts`
- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/metrics`

If your tenant uses different payloads/paths, update:
- `_upload_artifact`
- `_create_profile_job`
- `_wait_for_completion`
- `_extract_metrics`

in `src/edge_qnn_pipeline/profiling/aihub_client.py`.

## Expected Metrics Keys

Parser currently checks:
- latency: `latency_ms`, `avg_latency_ms`, `inference_time_ms`
- throughput: `throughput_fps`, `fps`
- power: `power_watts`, `avg_power_w`
- memory: `memory_mb`, `peak_memory_mb`

Adjust `_extract_optional_metric` lookups for your report schema.

## Runtime Optimization Loop With Real API

1. Run with `dry_run: false` and `aihub.mode: qai_hub`.
2. Validate one iteration completes.
3. Confirm parsed latency matches AI Hub UI report.
4. Expand optimization policy only after metrics parsing is stable.

## Security

- Keep API key only in environment variables.
- Avoid writing bearer tokens in config or logs.
- Restrict project permissions to profiling scope.
