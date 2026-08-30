"""F6: a top-level STRING error body is treated as malformed (not preserved), so it
is replaced with the canonical upstream_protocol_error rather than leaking upstream
payload. Pins the deliberate dict-only contract."""
import pytest

from proxy.openai_compat import chat_completion_shape_error


def test_dict_error_without_choices_is_preserved():
    # object-valued error + no choices -> NOT malformed (None), preserved upstream.
    assert chat_completion_shape_error({"error": {"message": "x"}}) is None


def test_string_error_is_treated_as_malformed():
    # top-level string error -> falls through to shape checks -> malformed reason.
    reason = chat_completion_shape_error({"error": "provider rejected request"})
    assert reason is not None            # classified malformed
    assert "choices" in reason           # fails on the missing/!list choices check


def test_list_error_is_treated_as_malformed():
    assert chat_completion_shape_error({"error": ["boom"]}) is not None


def test_null_error_is_treated_as_malformed():
    assert chat_completion_shape_error({"error": None}) is not None


def test_error_dict_with_choices_is_still_malformed():
    # Even if error is a dict, presence of choices -> response IS the error body,
    # but shape_error_flag still returns the malformed reason because choices is
    # not a list / message is missing. (Pins the full-contract interaction.)
    reason = chat_completion_shape_error({"error": {"message": "x"}, "choices": "nope"})
    assert reason is not None
