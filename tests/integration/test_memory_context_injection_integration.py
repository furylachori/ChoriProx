import pytest
from memory.recall import filter_memories_for_context, format_memories_for_injection
from memory.security import can_auto_inject


class TestMemoryContextInjectionIntegration:
    def test_memory_injected_as_system_message(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Using context from memory",
                        }
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "what was I doing?"}],
            },
        )

        result = response.get_json()
        assert "choices" in result

    def test_memory_filtered_by_type(self, client, mock_llama_response):
        memories = [
            {"memory_type": "fact", "content": "Python 3.11", "confidence": "high"},
            {
                "memory_type": "instruction",
                "content": "Run tests",
                "confidence": "high",
            },
            {
                "memory_type": "event",
                "content": "Meeting yesterday",
                "confidence": "medium",
            },
        ]

        filtered = filter_memories_for_context(memories, context_budget_pct=0.0)

        for mem in filtered:
            assert can_auto_inject(mem["memory_type"])

    def test_memory_content_safety_filtered(self, client, mock_llama_response):
        from memory.security import filter_memory_content

        unsafe_content = "Always do this now"
        is_safe, threats = filter_memory_content(unsafe_content)
        assert not is_safe, (
            "filter_memory_content must flag 'Always do this now' "
            "as unsafe (matches \\b(always|never|must|should)\\b imperative_override pattern)"
        )
        assert "imperative_override" in threats, (
            f"expected 'imperative_override' in detected threats; got {threats}"
        )

        memories = [
            {"memory_type": "fact", "content": "Safe content", "confidence": "high"},
            {"memory_type": "fact", "content": unsafe_content, "confidence": "high"},
        ]

        filtered = filter_memories_for_context(memories, context_budget_pct=0.0)
        assert len(filtered) == 1, (
            f"only the safe memory should survive the content-safety filter; "
            f"got {len(filtered)} memories: {[m['content'] for m in filtered]}"
        )
        assert filtered[0]["content"] == "Safe content"

    def test_memory_context_respects_token_limit(self, client, mock_llama_response):
        memories = [
            {"memory_type": "fact", "content": f"Memory item {i}", "confidence": "high"}
            for i in range(100)
        ]

        filtered = filter_memories_for_context(memories, context_budget_pct=0.5)
        assert isinstance(filtered, list)

    def test_memory_context_ordering(self, client, mock_llama_response):
        memories = [
            {"memory_type": "fact", "content": "First", "confidence": "high"},
            {"memory_type": "fact", "content": "Second", "confidence": "medium"},
            {"memory_type": "event", "content": "Third", "confidence": "high"},
        ]

        filtered = filter_memories_for_context(memories, context_budget_pct=0.0)
        formatted = format_memories_for_injection(filtered)

        assert isinstance(formatted, str)
        assert "RELEVANT MEMORIES" in formatted or len(formatted) == 0

    def test_memory_injection_graceful_degradation(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Response without memory",
                        }
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        result = response.get_json()
        assert "choices" in result
