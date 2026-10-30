# Changelog

## v0.1.0 - 2026-10-30

- First release: tiny PyTorch CNN (4 synthetic shape classes), FastAPI serving, Prometheus metrics, JSON logs.
- Measured receipts in docs/bench.json: 1.000 validation accuracy, 0.179 ms p50 at batch 1, 47 kB checkpoint.
- Serving depends on a Predictor protocol, so the HTTP surface is tested without torch; torch is an optional extra behind one adapter.
- Degraded mode is labelled, never disguised: a missing checkpoint or missing torch reports `source: "stub"` on /model and /ready.
