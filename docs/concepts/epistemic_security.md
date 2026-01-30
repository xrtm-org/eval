# Epistemic Security

**Epistemic Security** is the defense against information hazards, hallucinations, and untrusted sources.

## The Trust Guardian

The `IntegrityGuardian` class is the gatekeeper for all information entering the evaluation logic.

```python
guardian = IntegrityGuardian(registry=my_registry)
guardian.verify_source("https://untrusted-news.com")
```

## Source Trust Registry

`xrtm-eval` maintains a registry of trusted domains (`SourceTrustRegistry`).
- **Whitelist**: Known high-integrity sources (e.g., Reuters, Nature).
- **Blacklist**: Known hallucination-prone or disinfo sources.

### Evaluation Impact
If a forecast relies on a blacklisted source in its `reasoning_trace`, the `EpistemicScore` is penalized, even if the probability was accurate.
