# Metrics & Observability

## Endpoints
| Endpoint | Description |
|----------|-------------|
| `/api/v1/nlp/metrics` | Prometheus exposition (embedding, dedupe, pipeline, autosolve, QUIC, Redis) |
| `/api/v1/pipeline/metrics` | JSON pipeline histograms + dedupe stats |
| `/api/v1/pipeline/recent-samples.csv` | CSV export of last 25 samples per stage |
| `/api/v1/quic/metrics` | QUIC rolling metrics (connections, streams, p50/p90/p99, error_rate_1m) |
| `/api/v1/quic/push` | POST ingestion endpoint for external QUIC poller (validated + rate limited) |
| `/api/v1/redis/metrics` | Redis health / latency |

## Poller Script
`npm run metrics:quic:poll` polls backend QUIC source (env `QUIC_SOURCE_URL`) and pushes samples to `/api/v1/quic/push`.

## Dashboard Features
- Sparkline recent samples
- Anomaly badges per stage (median-based outlier + negative delta)
- QUIC quantiles & 1m error count
- Pause/Resume auto refresh
- CSV download button

## Alert Thresholds (Environment)
```
QUIC_ALERT_P99_MS=800
QUIC_ALERT_ERRORS_1M=5
PIPELINE_ALERT_ANOMALIES_5M=20
```

## Aggregate Gauges
- `pipeline_latency_anomalies_last5m`

See `SLA_SLO_METRICS.md` for targets & error budgets.
