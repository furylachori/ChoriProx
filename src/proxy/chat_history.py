"""Chat history cache and LRU eviction.

Extracted from state.py to reduce the god object.
"""
import os
import threading
import logging
from collections import OrderedDict
from proxy.lock_ordering import make_lock

logger = logging.getLogger(__name__)

chat_histories_lock = make_lock(2, "chat_histories_lock", threading.RLock)  # Position #2 — DO NOT MOVE

CHAT_HISTORIES_MAX_ENTRIES = int(os.environ.get("CHAT_HISTORIES_MAX_ENTRIES", "100"))
_chat_histories_lru: OrderedDict = OrderedDict()


class ChatHistoryCache(dict):
    """LRU cache for chat histories backed by SQLite.

    Inherits from dict so all existing callers (get, setdefault, []=, pop, …)
    continue to work without any code changes.

    Behaviour
    ---------
    * **Reads**  – ``__getitem__`` / ``__contains__`` / ``get()`` update the
      LRU order on a hit.  ``__missing__`` fires on a ``KeyError`` miss and
      performs a read-through load from SQLite via ``_load_from_sqlite``.
    * **Writes** – ``__setitem__`` (triggered by ``chat_histories[k] = v`` and
      by ``__missing__`` after a DB load) adds the key to the LRU and evicts
      the oldest entry when the cache exceeds ``CHAT_HISTORIES_MAX_ENTRIES``.
    * **Eviction** – the least-recently-used session is evicted from both the
      dict and the LRU tracking structure; the next access reloads it from
      SQLite transparently.
    * **Thread-safety** – ``_track_access`` and ``_evict`` are guarded by
      ``chat_histories_lock`` (an ``RLock`` so callers that already hold the
      lock – e.g. ``reload_session_from_db`` – do not deadlock).
    """

    _MISSING = object()

    def __getitem__(self, key):
        """Return value for *key*, updating LRU on a cache hit."""
        value = super().__getitem__(key)
        with chat_histories_lock:
            evicted_ids = _track_access(key)
        from proxy.eviction_coordinator import build_and_process_deferred_clears
        build_and_process_deferred_clears(evicted_ids)
        return value

    def __missing__(self, key):
        """Read-through: load *key* from SQLite on a cache miss."""
        with chat_histories_lock:
            value = _load_from_sqlite(key)
            if value is None:
                raise KeyError(key)
            evicted_ids = _track_access(key)
            super().__setitem__(key, value)
        from proxy.eviction_coordinator import build_and_process_deferred_clears
        build_and_process_deferred_clears(evicted_ids)
        return value

    def __setitem__(self, key, value):
        """Store *value* under *key* and update LRU order."""
        super().__setitem__(key, value)
        with chat_histories_lock:
            evicted_ids = _track_access(key)
        from proxy.eviction_coordinator import build_and_process_deferred_clears
        build_and_process_deferred_clears(evicted_ids)

    def __contains__(self, key):
        return super().__contains__(key)

    def get(self, key, default=_MISSING):
        """Return *key* or *default*, loading from SQLite on a miss.

        Unlike the standard ``dict.get`` this method also triggers the
        read-through path (``__missing__``) when *key* is absent, so callers
        never need an explicit ``reload_session_from_db`` round-trip.
        """
        if key in self:
            with chat_histories_lock:
                evicted_ids = _track_access(key)
            from proxy.eviction_coordinator import build_and_process_deferred_clears
            build_and_process_deferred_clears(evicted_ids)
            try:
                return super().__getitem__(key)
            except KeyError:
                pass
        try:
            return self[key]
        except KeyError:
            if default is not self._MISSING:
                return default
            raise

    def pop(self, key, *args):
        """Remove *key* and return its value, cleaning up LRU tracking."""
        with chat_histories_lock:
            _chat_histories_lru.pop(key, None)
        return super().pop(key, *args)


def _track_access(session_id: str) -> list:
    """Touch *session_id* in the LRU and evict if the cache is over capacity.

    Must be called while ``chat_histories_lock`` is held (or from inside
    ``ChatHistoryCache`` methods where the lock is already re-entrantly
    acquired via ``RLock``).

    Returns a list of evicted session_ids. Pass them to
    ``build_and_process_deferred_clears`` AFTER releasing ``chat_histories_lock``
    to avoid lock-ordering deadlocks.
    """
    if session_id in _chat_histories_lru:
        _chat_histories_lru.move_to_end(session_id)
    else:
        _chat_histories_lru[session_id] = None

    return _evict_chat_histories_lru()


def _evict_chat_histories_lru() -> list:
    """Evict the least-recently-used session when the cache is over capacity.

    Must be called while ``chat_histories_lock`` is held.

    Returns a list of evicted session_ids. The caller must pass these to
    ``build_and_process_deferred_clears`` AFTER releasing ``chat_histories_lock``
    to prevent lock-ordering deadlocks (AgentRegistry._lock is position 0,
    chat_histories_lock is position 2 — never acquire #0 while holding #2).
    """
    evicted_session_ids = []
    while len(_chat_histories_lru) > CHAT_HISTORIES_MAX_ENTRIES:
        session_id, _ = _chat_histories_lru.popitem(last=False)
        dict.pop(chat_histories, session_id, None)
        evicted_session_ids.append(session_id)
        logger.info(
            "Evicted session %s from chat_histories LRU (max=%d)",
            session_id[:8],
            CHAT_HISTORIES_MAX_ENTRIES,
        )
    return evicted_session_ids


def _load_from_sqlite(session_id: str):
    """Load chat history for *session_id* from SQLite.

    Returns a list of messages, or ``None`` if the session has no stored
    history.
    """
    try:
        from memory.db import get_last_messages, session_exists

        if not session_exists(session_id):
            return None

        messages = get_last_messages(session_id, n=500)
        return messages
    except Exception:
        logger.warning(
            "chat_histories LRU: failed to load session %s from SQLite",
            session_id,
            exc_info=True,
        )
        return None


chat_histories = ChatHistoryCache()
