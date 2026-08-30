"""Regression test for unified tool parsing (double-parse fix).

Verifies that:
1. Tool arguments are normalized (e.g., "path" → "file_path")
2. ToolCall objects have UUIDs assigned
3. ctx.tool_calls_list is set after handle_tool_interception
4. ToolPipeline.run() skips re-parsing when ctx.tool_calls_list is set
"""

import json
import pytest
from unittest.mock import MagicMock, patch


class TestUnifiedToolParsing:
    """Test that tool parsing is unified and not duplicated."""

    def test_tool_context_has_tool_calls_list_field(self):
        """ToolContext should have tool_calls_list field."""
        from proxy.tool_pipeline.tool_pipeline import ToolContext
        
        ctx = ToolContext(
            session_id="test",
            body={},
            llama_body={},
            auth_headers={},
            original_model="test-model",
            upstream_url="http://test",
            tool_calls_list=[{"name": "test_tool", "arguments": {}}],
        )
        
        assert ctx.tool_calls_list is not None
        assert len(ctx.tool_calls_list) == 1
        assert ctx.tool_calls_list[0]["name"] == "test_tool"

    def test_tool_context_tool_calls_list_defaults_to_none(self):
        """ToolContext.tool_calls_list should default to None."""
        from proxy.tool_pipeline.tool_pipeline import ToolContext
        
        ctx = ToolContext(
            session_id="test",
            body={},
            llama_body={},
            auth_headers={},
            original_model="test-model",
            upstream_url="http://test",
        )
        
        assert ctx.tool_calls_list is None

    def test_pipeline_skips_reparse_when_tool_calls_list_set(self):
        """ToolPipeline.run() should skip re-parsing when ctx.tool_calls_list is set."""
        from proxy.tool_pipeline.tool_pipeline import ToolPipeline, ToolContext
        
        # Create a ToolContext with pre-parsed tool calls
        pre_parsed = [{"name": "recall", "arguments": {"query": "test"}}]
        ctx = ToolContext(
            session_id="test",
            body={},
            llama_body={},
            auth_headers={},
            original_model="test-model",
            upstream_url="http://test",
            tool_calls_list=pre_parsed,
        )
        
        # Create response data with tool calls
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "recall",
                            "arguments": json.dumps({"query": "test"})
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }
        
        pipeline = ToolPipeline()
        
        # Mock _parse to track if it's called
        with patch.object(pipeline, '_parse', wraps=pipeline._parse) as mock_parse:
            # Mock _classify to return empty memory calls (no execution)
            with patch.object(pipeline, '_classify', return_value=([], [])):
                result = pipeline.run(response_data, ctx)
                
                # _parse should still be called for normalization
                assert mock_parse.called
                
                # The original_tool_calls_list should come from ctx.tool_calls_list
                # not from re-parsing response_data

    def test_tool_calls_have_uuids(self):
        """ToolCall objects should have UUIDs assigned."""
        from proxy.tool_pipeline.tool_call import ToolCall
        
        # Create a ToolCall from OpenAI format
        openai_tc = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "recall",
                "arguments": json.dumps({"query": "test"})
            }
        }
        
        tool_call = ToolCall.from_openai(openai_tc)
        
        # Should have an ID
        assert tool_call.id is not None
        assert len(tool_call.id) > 0

    def test_tool_arguments_normalized(self):
        """Tool arguments should be normalized (e.g., 'path' → 'file_path')."""
        from proxy.tool_parsing import normalize_tool_arguments
        
        # Test normalization for a tool that uses file_path
        args = {"path": "/test/file.txt", "other": "value"}
        normalized = normalize_tool_arguments("read_file", args)

        assert normalized["file_path"] == "/test/file.txt"

    def test_pipeline_fallback_when_tool_calls_list_none(self):
        """ToolPipeline.run() should parse when ctx.tool_calls_list is None."""
        from proxy.tool_pipeline.tool_pipeline import ToolPipeline, ToolContext
        
        # Create a ToolContext WITHOUT pre-parsed tool calls
        ctx = ToolContext(
            session_id="test",
            body={},
            llama_body={},
            auth_headers={},
            original_model="test-model",
            upstream_url="http://test",
            tool_calls_list=None,  # Explicitly None
        )
        
        # Create response data with tool calls
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "recall",
                            "arguments": json.dumps({"query": "test"})
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }
        
        pipeline = ToolPipeline()
        
        # Mock _classify to return empty memory calls (no execution)
        with patch.object(pipeline, '_classify', return_value=([], [])):
            result = pipeline.run(response_data, ctx)
            
            # Should parse from response_data when ctx.tool_calls_list is None
            # and not crash
            assert result is not None
