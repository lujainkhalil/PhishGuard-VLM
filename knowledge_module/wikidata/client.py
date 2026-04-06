"""
Wikidata SPARQL + entity search client with disk cache, retries, and connection reuse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

DEFAULT_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
DEFAULT_USER_AGENT = "Phishguard-VLM/1.0 (Wikidata SPARQL client; research/education)"


@dataclass
class BrandInfo:
    """Structured brand row from Wikidata (search + SPARQL enrichment)."""

    qid: str
    label: str
    description: str | None
    official_websites: list[str]
    logo_urls: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WikidataClient:
    """
    Query Wikidata for brand-like entities: SPARQL execution, entity search, local JSON cache.

    - **Latency**: cache hits avoid the network; :class:`requests.Session` keeps connections warm.
    - **Failures**: timeouts and 5xx/429 trigger limited retries with backoff; failures return
      ``None`` / empty bindings instead of raising (see method docs).
    """

    def __init__(
        self,
        *,
        sparql_endpoint: str = DEFAULT_SPARQL_ENDPOINT,
        api_endpoint: str = DEFAULT_API_ENDPOINT,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: int = 86_400,
        timeout: tuple[float, float] = (3.0, 12.0),
        max_retries: int = 3,
        retry_backoff_base: float = 0.75,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.sparql_endpoint = sparql_endpoint.rstrip("/")
        self.api_endpoint = api_endpoint.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> WikidataClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- cache ---

    def _cache_key(self, namespace: str, payload: str) -> str:
        h = hashlib.sha256(f"{namespace}:{payload}".encode("utf-8")).hexdigest()[:32]
        return h

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if path is None or not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            ts = float(raw.get("cached_at", 0))
            if self.cache_ttl_seconds > 0 and (time.time() - ts) > self.cache_ttl_seconds:
                return None
            return raw.get("data")
        except Exception as e:
            logger.debug("Cache read failed %s: %s", path, e)
            return None

    def _cache_set(self, key: str, data: Any) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps({"cached_at": time.time(), "data": data}, ensure_ascii=False, indent=0),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Cache write failed %s: %s", path, e)

    # --- HTTP ---

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        acceptable_status: tuple[int, ...] = (200,),
    ) -> requests.Response | None:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._session.request(
                    method,
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers=headers,
                )
                if r.status_code in acceptable_status:
                    return r
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", self.retry_backoff_base * (2**attempt)))
                    logger.warning("Wikidata rate limited (429); sleeping %.1fs", wait)
                    time.sleep(min(wait, 60.0))
                    continue
                if 500 <= r.status_code < 600:
                    logger.warning("Wikidata server error %s (attempt %d)", r.status_code, attempt + 1)
                    time.sleep(self.retry_backoff_base * (2**attempt))
                    continue
                logger.warning("Wikidata HTTP %s for %s", r.status_code, url[:80])
                return None
            except requests.RequestException as e:
                last_exc = e
                logger.warning("Wikidata request failed (attempt %d): %s", attempt + 1, e)
                time.sleep(self.retry_backoff_base * (2**attempt))
        if last_exc:
            logger.error("Wikidata request gave up after retries: %s", last_exc)
        return None

    def sparql(self, query: str, *, use_cache: bool = True) -> dict[str, Any] | None:
        """
        Run a SPARQL query and return parsed JSON results (``results.bindings`` structure),
        or ``None`` on failure.

        Response shape matches ``application/sparql-results+json`` (``head`` + ``results``).
        """
        query = query.strip()
        cache_key = self._cache_key("sparql", query)
        if use_cache and self.cache_dir:
            hit = self._cache_get(cache_key)
            if hit is not None:
                return hit

        r = self._request(
            "GET",
            self.sparql_endpoint,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            acceptable_status=(200,),
        )
        if r is None:
            return None
        try:
            data = r.json()
        except json.JSONDecodeError as e:
            logger.error("Invalid SPARQL JSON: %s", e)
            return None

        if use_cache and self.cache_dir:
            self._cache_set(cache_key, data)
        return data

    def sparql_bindings(self, query: str, *, use_cache: bool = True) -> list[dict[str, Any]]:
        """Convenience: return the list of binding dicts (variable -> ``{type, value}``)."""
        raw = self.sparql(query, use_cache=use_cache)
        if not raw or "results" not in raw:
            return []
        return list(raw["results"].get("bindings") or [])

    def search_entities(
        self,
        text: str,
        *,
        language: str = "en",
        limit: int = 10,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Call ``wbsearchentities`` (fast MediaWiki API). Returns list of hit dicts with ``id``, ``label``, etc.
        On failure returns [].
        """
        text = (text or "").strip()
        if not text:
            return []
        cache_key = self._cache_key("search", f"{language}:{limit}:{text.lower()}")
        if use_cache and self.cache_dir:
            hit = self._cache_get(cache_key)
            if hit is not None:
                return hit

        params = {
            "action": "wbsearchentities",
            "search": text,
            "language": language,
            "format": "json",
            "limit": min(limit, 50),
        }
        r = self._request("GET", self.api_endpoint, params=params, acceptable_status=(200,))
        if r is None:
            return []
        try:
            payload = r.json()
        except json.JSONDecodeError:
            return []
        out = payload.get("search") or []
        if use_cache and self.cache_dir:
            self._cache_set(cache_key, out)
        return out

    def get_brand_info(
        self,
        brand_name: str,
        *,
        language: str = "en",
        search_limit: int = 5,
        use_cache: bool = True,
    ) -> BrandInfo | None:
        """
        Resolve a brand string to Wikidata: search entities, take the best match, fetch P856 (website) and P154 (logo).

        Returns ``None`` if search fails or enrichment fails. API/SPARQL errors are logged and swallowed
        to keep callers simple.
        """
        brand_name = (brand_name or "").strip()
        if not brand_name:
            return None

        cache_key = self._cache_key("brand", f"{language}:{search_limit}:{brand_name.lower()}")
        if use_cache and self.cache_dir:
            hit = self._cache_get(cache_key)
            if hit is not None:
                try:
                    return BrandInfo(**hit)
                except TypeError:
                    pass

        hits = self.search_entities(brand_name, language=language, limit=search_limit, use_cache=use_cache)
        if not hits:
            return None
        top = hits[0]
        qid = top.get("id")
        if not qid or not re.match(r"^Q\d+$", qid):
            return None

        label = (top.get("label") or "").strip() or qid
        desc = top.get("description")

        sites: set[str] = set()
        logos: set[str] = set()
        bindings = self._bindings_for_item(qid)
        for b in bindings:
            w = _binding_value(b.get("website"))
            if w:
                sites.add(w)
            lg = _binding_value(b.get("logo"))
            if lg:
                logos.add(_commons_file_url(lg) or lg)

        info = BrandInfo(
            qid=qid,
            label=label,
            description=desc,
            official_websites=sorted(sites),
            logo_urls=sorted(logos),
        )
        if use_cache and self.cache_dir:
            self._cache_set(cache_key, info.to_dict())
        return info

    def _bindings_for_item(self, qid: str) -> list[dict[str, Any]]:
        """SPARQL OPTIONAL rows for P856 and P154."""
        safe_qid = qid if re.match(r"^Q\d+$", qid) else "Q0"
        query = f"""
SELECT ?website ?logo WHERE {{
  BIND(wd:{safe_qid} AS ?item)
  OPTIONAL {{ ?item wdt:P856 ?website . }}
  OPTIONAL {{ ?item wdt:P154 ?logo . }}
}}
""".strip()
        return self.sparql_bindings(query, use_cache=True)


def _binding_value(cell: dict[str, Any] | None) -> str | None:
    if not cell or "value" not in cell:
        return None
    return str(cell["value"]).strip() or None


def _commons_file_url(value: str) -> str | None:
    """Normalize Wikidata/commons file IRIs to a stable Special:FilePath URL when possible."""
    if not value:
        return None
    if value.startswith("http://commons.wikimedia.org/wiki/Special:FilePath/"):
        return "https://commons.wikimedia.org/wiki/Special:FilePath/" + quote(
            value.split("/Special:FilePath/", 1)[-1], safe=""
        )
    if value.startswith("http://commons.wikimedia.org/wiki/File:"):
        name = value.rsplit("/File:", 1)[-1]
        return "https://commons.wikimedia.org/wiki/Special:FilePath/" + quote(name, safe="")
    if value.startswith("https://commons.wikimedia.org/wiki/File:"):
        name = value.rsplit("/File:", 1)[-1]
        return "https://commons.wikimedia.org/wiki/Special:FilePath/" + quote(name, safe="")
    return value
