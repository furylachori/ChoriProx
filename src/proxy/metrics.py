"""Prometheus metrics for ChoriProx observability.

Enable with PROXY_METRICS_ENABLED=1 environment variable.
Scrape from /metrics endpoint.
"""
import os
import logging
import threading

logger = logging.getLogger(__name__)

# Module-init lock, intentionally outside state.py hierarchy.
# Used once per process; not request-scoped.
_metrics_init_lock = threading.Lock()

_METRICS_ENABLED = os.environ.get("PROXY_METRICS_ENABLED", "").lower() in ("1", "true", "yes")

_prom = None

def _get_prom():
    global _prom
    if _prom is None and _METRICS_ENABLED:
        try:
            import prometheus_client
            _prom = prometheus_client
        except ImportError:
            logger.warning("prometheus_client not installed — metrics disabled")
            _prom = False
    return _prom


def is_enabled():
    return _METRICS_ENABLED and _get_prom() is not False


_http_requests_total = None
_http_request_duration = None
_rate_limiter_checks_total = None
_rate_limiter_rejections_total = None
_neo4j_queries_total = None
_neo4j_query_duration = None
_neo4j_errors_total = None
_agent_registry_evictions_total = None
_prompt_cache_requests_total = None
_prompt_cache_read_tokens_total = None
_prompt_cache_no_cache_tokens_total = None


def _ensure_metrics():
    """Create metric objects on first use.

    M8-2: double-checked locking. The fast path is a non-atomic
    read of `_http_requests_total` — this is intentional and safe
    because Python attribute assignment is atomic under the GIL.
    The slow path (acquiring the lock + re-checking) runs only on
    the FIRST race. After the first thread wins, every subsequent
    thread takes the fast path.
    """
    # Fast path: already initialized (this read is atomic in CPython).
    if _http_requests_total is not None:
        return
    # Slow path: serialize concurrent first-callers via the module
    # init lock. Re-check under the lock to avoid duplicate work if
    # another thread raced ahead and finished initialization.
    with _metrics_init_lock:
        if _http_requests_total is not None:
            return
        if not is_enabled():
            return

        prom = _get_prom()
        _prom_module = prom

        globals()['_http_requests_total'] = _prom_module.Counter(
            'choriprox_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        globals()['_http_request_duration'] = _prom_module.Histogram(
            'choriprox_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        globals()['_rate_limiter_checks_total'] = _prom_module.Counter(
            'choriprox_rate_limiter_checks_total',
            'Rate limiter checks',
            ['limiter']
        )
        globals()['_rate_limiter_rejections_total'] = _prom_module.Counter(
            'choriprox_rate_limiter_rejections_total',
            'Rate limiter rejections',
            ['limiter']
        )
        globals()['_neo4j_queries_total'] = _prom_module.Counter(
            'choriprox_neo4j_queries_total',
            'Neo4j queries',
            ['operation']
        )
        globals()['_neo4j_query_duration'] = _prom_module.Histogram(
            'choriprox_neo4j_query_duration_seconds',
            'Neo4j query duration',
            ['operation'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        globals()['_neo4j_errors_total'] = _prom_module.Counter(
            'choriprox_neo4j_errors_total',
            'Neo4j errors',
            ['error_type']
        )
        globals()['_agent_registry_evictions_total'] = _prom_module.Counter(
            'choriprox_agent_registry_evictions_total',
            'Agent registry evictions'
        )
        globals()['_prompt_cache_requests_total'] = _prom_module.Counter(
            'choriprox_prompt_cache_requests_total',
            'Prompt cache requests',
            ['model', 'provider', 'cache_available']
        )
        globals()['_prompt_cache_read_tokens_total'] = _prom_module.Counter(
            'choriprox_prompt_cache_read_tokens_total',
            'Prompt cache read tokens',
            ['model', 'provider']
        )
        globals()['_prompt_cache_no_cache_tokens_total'] = _prom_module.Counter(
            'choriprox_prompt_cache_no_cache_tokens_total',
            'Prompt cache no-cache tokens',
            ['model', 'provider']
        )


def record_http_request(method, endpoint, status, duration):
    if not is_enabled():
        return
    _ensure_metrics()
    _http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    _http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

def record_rate_limiter_check(limiter_name):
    if not is_enabled():
        return
    _ensure_metrics()
    _rate_limiter_checks_total.labels(limiter=limiter_name).inc()

def record_rate_limiter_rejection(limiter_name):
    if not is_enabled():
        return
    _ensure_metrics()
    _rate_limiter_rejections_total.labels(limiter=limiter_name).inc()

def record_neo4j_query(operation, duration):
    if not is_enabled():
        return
    _ensure_metrics()
    _neo4j_queries_total.labels(operation=operation).inc()
    _neo4j_query_duration.labels(operation=operation).observe(duration)

def record_neo4j_error(error_type):
    if not is_enabled():
        return
    _ensure_metrics()
    _neo4j_errors_total.labels(error_type=error_type).inc()

def record_agent_eviction():
    if not is_enabled():
        return
    _ensure_metrics()
    _agent_registry_evictions_total.inc()

def record_prompt_cache_metrics(model, provider, cache_read_tokens, no_cache_tokens):
    """Record prompt cache metrics from upstream response.
    
    Args:
        model: Model name string.
        provider: Provider name string (e.g. 'deepseek', 'qwen').
        cache_read_tokens: Number of cache-hit tokens, or None if unavailable.
        no_cache_tokens: Number of non-cached tokens, or None if unavailable.
    """
    if not is_enabled():
        return
    _ensure_metrics()

    cache_available = "true" if (cache_read_tokens is not None) else "false"
    _prompt_cache_requests_total.labels(
        model=model, provider=provider, cache_available=cache_available
    ).inc()

    if cache_read_tokens is not None:
        _prompt_cache_read_tokens_total.labels(model=model, provider=provider).inc(cache_read_tokens)

    if no_cache_tokens is not None:
        _prompt_cache_no_cache_tokens_total.labels(model=model, provider=provider).inc(no_cache_tokens)


def generate_metrics():
    """Generate Prometheus metrics output for /metrics endpoint."""
    if not is_enabled():
        return "", 404
    prom = _get_prom()
    return prom.generate_latest(), 200, {'Content-Type': prom.CONTENT_TYPE_LATEST}
