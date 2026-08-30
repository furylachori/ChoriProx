

def test_extract_text_content_multi_part_join():
    """Pin the multi-part join semantic. Canonical uses ' ' (space)
    as joiner between text parts. Previously dispatch.py used '' (empty)
    which produced different output for multi-part content."""
    from proxy.utils import extract_text_content

    parts = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]
    result = extract_text_content(parts)
    assert result == "hello world", (
        f"Expected 'hello world' (space-joined), got {result!r}. "
        f"The canonical joiner is ' ' — if this changes, dispatch.py callers "
        f"will produce different output than expected."
    )


def test_extract_text_content_single_part():
    """Single text part returns the text directly (no joiner)."""
    from proxy.utils import extract_text_content

    parts = [{"type": "text", "text": "only"}]
    assert extract_text_content(parts) == "only"


def test_extract_text_content_image_only():
    """Image-only content returns '[image]' placeholder (not empty)."""
    from proxy.utils import extract_text_content

    parts = [{"type": "image_url", "image_url": {"url": "http://x"}}]
    result = extract_text_content(parts)
    assert result == "[image]", (
        f"Image-only content should produce '[image]' placeholder, got {result!r}"
    )
