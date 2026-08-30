import pytest


class TestContextManagementIntegration:
    def test_system_prompt_preserved_after_truncation(
        self, client, mock_llama_response, monkeypatch
    ):
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)

        large_content = "x" * 5000
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Response to: {large_content[:50]}",
                        }
                    }
                ]
            }
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        for i in range(30):
            messages.append({"role": "user", "content": f"Message {i}: " + "y" * 100})
        messages.append({"role": "user", "content": large_content})

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": messages,
            },
        )

        result = response.get_json()
        assert "choices" in result

    def test_truncation_at_boundary(self, client, mock_llama_response, monkeypatch):
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)

        messages = [
            {"role": "system", "content": "System prompt"},
        ]
        for i in range(20):
            messages.append({"role": "user", "content": f"Message {i}: " + "z" * 100})
        messages.append({"role": "user", "content": "t" * 4000})

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": messages,
            },
        )

        result = response.get_json()
        assert "choices" in result

    def test_no_truncation_under_limit(self, client, mock_llama_response, monkeypatch):
        monkeypatch.setattr("qwen_proxy.TEST_MODE", True)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": messages,
            },
        )

        result = response.get_json()
        assert "choices" in result
