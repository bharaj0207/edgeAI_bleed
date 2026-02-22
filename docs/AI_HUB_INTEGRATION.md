# AI Hub Integration Notes

## Recommended Mode (Official SDK)

The default real profile path uses Qualcomm's official `qai_hub` Python SDK.

References:
- [AI Hub Get Started](https://aihub.qualcomm.com/get-started)
- [API Docs](https://app.aihub.qualcomm.com/docs/hub/api.html)
- [submit_profile_job](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_profile_job.html)
- [ProfileJob.download_profile](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.client.ProfileJob.html)

### Required Inputs

- `QAI_HUB_API_KEY` environment variable
- `aihub.project`
- `aihub.device`

### Config Example

```yaml
aihub:
  mode: qai_hub_sdk
  api_key_env: QAI_HUB_API_KEY
  project: edge-optimization
  device: Samsung Galaxy S24 (Family)
  profile_options: --compute_unit npu --qairt_version default --target_runtime qnn_context_binary
```

### Runtime Flow

1. `qai_hub.upload_model(<local-context-binary>)`
2. `qai_hub.submit_profile_job(...)`
3. `ProfileJob.wait(...)`
4. `ProfileJob.download_profile()`
5. parse latency/throughput/power/memory from profile payload

## REST Fallback Mode

Use only if your deployment requires custom HTTP endpoints.

```yaml
aihub:
  mode: qai_hub_rest
  base_url: https://<your-ai-hub-endpoint>
```

REST fallback implementation is in:
- `/Users/bharadwaj/Documents/model_creation/src/edge_qnn_pipeline/profiling/aihub_client.py`

## API Token Setup

SDK mode supports either:
- CLI configuration (`qai-hub configure --api_token ...`), or
- runtime token from environment via `aihub.api_key_env` (pipeline calls `qai_hub.set_session_token(...)` when present)

## Validation Checklist

1. Run one iteration with `dry_run: false`.
2. Confirm profile job status is `SUCCESS`.
3. Compare parsed latency in `profile_metrics.json` against AI Hub report.
4. Then enable full optimization loop.

## Security

- Keep API key in environment only.
- Do not commit tokens, profile credentials, or private endpoint secrets.
- Scope token permissions to minimal required access.
