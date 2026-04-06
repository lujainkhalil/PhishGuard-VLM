# Knowledge Module

Brand verification via Wikidata to support impersonation detection and zero-shot brands.

## Subdirectories

| Directory | Responsibility |
|-----------|-----------------|
| **wikidata/** | SPARQL client for official domains, aliases, logos; response caching (in-memory or Redis). |
| **brand_matching/** | Compare page domain to claimed brand’s official domain(s); return match/mismatch/unknown. |

## Design

- Brand names come from VLM output or a dedicated extractor.
- Caching is essential for latency and to avoid rate limits; TTL and size in config.
- System can run without this module (ablation); aggregator uses result only when available.
