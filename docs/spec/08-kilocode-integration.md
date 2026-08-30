# Component: Kilocode Integration

> Part of [SPEC](../SPEC.md). See also [Implementation](../implementation/08-kilocode-integration.md)

## Setup

1. Start the Qwen3-Coder model on `localhost:7777`.
2. Start the proxy: `python3 qwen_proxy.py` or `./start_qwencode.sh`.

## Config Files

- `.kilo/agent/qwencode.json` — agent definition.
- `.kilo/command/qwencode.md` — command documentation.
- `kilo.json` — main configuration.

## Usage

```bash
/qwencode "Read the README.md file and summarize it"
```

The proxy parses Qwen's tool calls (XML/JSON), executes them locally, and returns results.

## Tools

`read_file`, `write_file`, `edit_file`, `file_glob`, `run_shell_command`, `web_fetch`.

## Testing

`python3 test_proxy.py` verifies tool execution and writes output to `test_proxy_output.txt`.
