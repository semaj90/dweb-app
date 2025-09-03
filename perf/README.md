Performance Metrics Directory
=============================

Purpose: Holds baseline and historical build performance artifacts plus schema.

Files:
 - baseline.json            -> Initial accepted clean build metrics (edit after first run)
 - perf_schema.json         -> Schema documentation for metric objects
 - build-<git-sha>/         -> Per-run directory containing summary.json and any trace files

Metric Definitions (Phase 1):
 - clean_ms: Full clean build wall-clock (median across repeats)
 - incremental_ms: Incremental build time after touching a small translation unit
 - link_ms: Link stage duration (if extractable)
 - cache_hit_ratio: From compiler cache tool (sccache/ccache) if available, else null
 - template_instantiations: Count parsed from trace or compiler stats (optional placeholder now)
 - hdr_top: Array of {header, parse_ms, pct_total}

Workflow (Initial):
 1. Run scripts/perf/measure_build.py --clean --repeat 3
 2. Commit generated baseline.json (rename summary to baseline on first run)
 3. CI compares new clean_ms/link_ms against baseline (+5% / +8% thresholds) once gating enabled.
