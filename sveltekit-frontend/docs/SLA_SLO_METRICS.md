# Metrics SLA / SLO

| Domain | Metric | Target (p50) | Target (p90) | Target (p99) | Notes |
|--------|--------|--------------|--------------|--------------|-------|
| Pipeline | gpu stage latency | <15ms | <40ms | <120ms | Warm GPU path |
| Pipeline | wasm stage latency | <25ms | <60ms | <180ms | Fallback path |
| Pipeline | embedding stage latency | <80ms | <150ms | <400ms | Local model |
| Pipeline | retrieval stage latency | <30ms | <70ms | <200ms | Vector + filters |
| Pipeline | llm stage latency | <1500ms | <3000ms | <6000ms | Short responses |
| Pipeline | final stage latency | <50ms | <120ms | <300ms | Aggregation |
| QUIC | stream latency | <40ms | <120ms | <300ms | Internal LAN |
| Redis | ping latency | <3ms | <7ms | <20ms | Local instance |

## Error Budgets
- QUIC p99 > target threshold: budget 60 seconds / hour.
- Pipeline anomaly spikes (>20 anomalies/5m): budget 12 spikes / day.

## Alert Threshold Environment Variables
```
QUIC_ALERT_P99_MS=800
QUIC_ALERT_ERRORS_1M=5
PIPELINE_ALERT_ANOMALIES_5M=20
```

## Aggregate Gauges
- `pipeline_latency_anomalies_last5m` (gauge) used for anomaly spike detection.

## Actions
1. P99 breach sustained 3 cycles -> investigate upstream (network/model load).
2. Error spike + anomaly spike -> trigger autosolve maintenance cycle.
