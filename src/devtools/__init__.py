"""DevTools — pytest-free interactive debugging harness for the proxy.

Public surface:
    from devtools.debug_session import debug_session, DebugSession, ProxyResponse

`debug_session()` is a context manager that wires the proxy to a mock LLM
backend so you can drive the full request pipeline from IPython or a
script without pytest, real network, or real upstream credentials.

    with debug_session(canned_responses=[...]) as c:
        r = c.chat(messages=[{"role": "user", "content": "hi"}])
        c.activate_ticket("sess-1", workspace="choriprox")
        c.inspect_agent("sess-1")

The canonical use is in `tests/integration/test_debug_session.py` (pytest)
and `tests/debug/*.py` (worked-example scratch scripts).
"""
