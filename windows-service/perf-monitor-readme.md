# LegalAI Performance Monitor Service

A lightweight Go microservice + Windows Service scripts to collect runtime performance metrics (goroutines, heap, GC, uptime) for optimization (caching, hot-spot detection, capacity planning).

## Features
- Exposes JSON runtime metrics at `GET /metrics/runtime`
- Prometheus metrics exposition at `GET /metrics`
- Function argument signature frequency API (`POST /metrics/signature`, `GET /metrics/signatures`)
- Health endpoint at `GET /health`
- Periodic structured log emission (`METRIC {json}` every 30s)
- Tracks custom counters (requests, errors, events)
- Windows Service install/uninstall scripts
- Batch collector to archive periodic snapshots

## Install (Windows Service)
```powershell
cd windows-service
./perf-monitor-install.bat
```
Service: `LegalAIPerfMonitor` (auto-start)

Uninstall:
```powershell
./perf-monitor-uninstall.bat
```

## Manual Run (Dev)
```powershell
go run ./go-microservice/cmd/perf-monitor
```
Then: http://localhost:8098/metrics/runtime

## Example JSON
```json
{
  "num_goroutine": 23,
  "heap_alloc": 3123456,
  "gc_count": 2,
  "cpu_percent": 4.3,
  "uptime_seconds": 120.4
}
```

## Batch Snapshot Collector
Continuous polling to `logs/perf/*.json`:
```powershell
./perf-monitor-collector.bat
```

## Next Enhancements
- Export Prometheus format
- Add endpoint latency timers via middleware wrappers
- Introduce argument signature hashing cache helper
- Push alerts (high GC pause, heap growth rate)
- Persist to Postgres / send to ELK stack

## Implemented Alert Thresholds
Environment variables (defaults in parentheses):
- `PERF_HEAP_WARN_MB` (800) / `PERF_HEAP_CRIT_MB` (1200)
- `PERF_GC_PAUSE_WARN_MS` (200) / `PERF_GC_PAUSE_CRIT_MS` (400)
- `PERF_CPU_WARN_PCT` (85) / `PERF_CPU_CRIT_PCT` (95)

## Caching Insights
POST collected signatures to `/metrics/signature` with body:
```json
{ "fn": "searchVectors", "argsHash": "ab12cd", "signature": "searchVectors:ab12cd" }
```
Top frequencies: `GET /metrics/signatures?limit=25`

