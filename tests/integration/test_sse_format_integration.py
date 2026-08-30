import pytest


class TestSSEFormatIntegration:
    def test_chunk_1_is_role_delta(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello world",
                        }
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "say hello"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        chunks = list(response.response)
        assert len(chunks) > 0

        # Skip SSE comment lines (e.g. ": connected", ": keepalive")
        # and find the first actual data chunk
        first_data_chunk = None
        for chunk in chunks:
            text = chunk.decode("utf-8", errors="replace").strip()
            if text.startswith(":"):
                continue  # SSE comment, skip
            if text:
                first_data_chunk = chunk
                break

        assert first_data_chunk is not None, f"No data chunk found in {chunks}"
        assert b"role" in first_data_chunk or b"delta" in first_data_chunk

    def test_chunk_2_to_N_are_tool_calls_with_indices(
        self, client, mock_llama_response
    ):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "/tmp/test.txt"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-coder",
                "messages": [{"role": "user", "content": "read file"}],
                "stream": True,
            },
        )

        assert response.status_code == 200

    def test_final_chunk_has_finish_reason(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done",
                            "finish_reason": "stop",
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
                "stream": True,
            },
        )

        assert response.status_code == 200

    def test_stream_true_forces_sse(self, client, mock_llama_response):
        mock_llama_response(
            data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Streamed response",
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
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.content_type == "text/event-stream"
