"""Drop into IPython with a debug_session pre-configured.

Run: `PYTHONPATH=src python -m devtools.shell`

Falls back to stdlib `code.interact()` when IPython is not installed.
"""
from __future__ import annotations

from devtools.debug_session import debug_session


def main() -> None:
    banner = (
        "DebugSession ready. `c` is the session.\n"
        "Try: c.chat(messages=[{'role':'user','content':'hi'}])\n"
        "     c.activate_ticket('my-session', workspace='choriprox')\n"
        "     c.inspect_agent('my-session')\n"
        "     c.dump_captured()\n"
        "Type `c.reset()` to clear state. Ctrl-D to exit.\n"
    )
    with debug_session() as c:
        try:
            from IPython import start_ipython

            start_ipython(argv=[], user_ns={"c": c}, banner1=banner)
        except ImportError:
            import code

            code.interact(banner=banner, local={"c": c})


if __name__ == "__main__":
    main()
