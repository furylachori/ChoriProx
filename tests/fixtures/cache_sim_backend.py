import json
from http.server import BaseHTTPRequestHandler


TOKENS_PER_CHAR = 0.25


class CacheSimulatingBackendHandler(BaseHTTPRequestHandler):
    _prev_messages_sig = None
    _cache_reads = None
    _totals = None
    _captured_raw_bodies: list = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        CacheSimulatingBackendHandler._captured_raw_bodies.append(raw)
        try:
            body = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            body = {}

        messages = body.get("messages", [])
        current_sig = [(m.get("role", ""), m.get("content", "")) for m in messages]

        prefix_len = 0
        prev = CacheSimulatingBackendHandler._prev_messages_sig
        if prev is not None:
            for i in range(min(len(current_sig), len(prev))):
                if current_sig[i] == prev[i]:
                    prefix_len = i + 1
                else:
                    break

        cache_read_tokens = 0.0
        for i in range(prefix_len):
            content = current_sig[i][1]
            cache_read_tokens += len(content) * TOKENS_PER_CHAR

        total_tokens = 0.0
        for role, content in current_sig:
            total_tokens += len(content) * TOKENS_PER_CHAR

        cache_read_int = int(cache_read_tokens)
        total_int = int(total_tokens)

        CacheSimulatingBackendHandler._cache_reads.append(cache_read_int)
        CacheSimulatingBackendHandler._totals.append(total_int)
        CacheSimulatingBackendHandler._prev_messages_sig = current_sig

        assistant_content = (
            "Understood. I have processed your request and here is a concise "
            "response summarizing the key points discussed so far in our "
            "conversation thread regarding the ongoing task at hand."
        )
        response_body = {
            "id": "chatcmpl-cache-sim",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": total_int,
                "completion_tokens": int(len(assistant_content) * TOKENS_PER_CHAR),
                "total_tokens": total_int + int(len(assistant_content) * TOKENS_PER_CHAR),
                "prompt_cache_hit_tokens": cache_read_int,
            },
        }

        data = json.dumps(response_body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def reset_cache_sim():
    CacheSimulatingBackendHandler._prev_messages_sig = None
    CacheSimulatingBackendHandler._cache_reads = []
    CacheSimulatingBackendHandler._totals = []
    CacheSimulatingBackendHandler._captured_raw_bodies = []


def steady_state_cache_read_pct(turns_to_drop=3):
    reads = CacheSimulatingBackendHandler._cache_reads
    totals = CacheSimulatingBackendHandler._totals
    n = len(reads)
    if n <= turns_to_drop:
        return 0.0
    window_reads = reads[turns_to_drop:]
    window_totals = totals[turns_to_drop:]
    total_read = sum(window_reads)
    total_total = sum(window_totals)
    if total_total == 0:
        return 0.0
    return (total_read / total_total) * 100.0
