import requests
import json
import subprocess
import os
import sys
import glob
from urllib import request, error

API_URL = "http://localhost:7777/v1/chat/completions"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "default": 1},
                    "limit": {"type": "integer", "default": 2000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (overwrites existing)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace text in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_glob",
            "description": "Find files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "root_dir": {"type": "string", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Execute a shell command. Provide the exact command you want to run in the 'command' parameter (e.g., 'ls -la', 'grep \"pattern\" file.txt', 'python3 script.py').",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch content from a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["url"],
            },
        },
    },
    # we can add the memory integration tools here (save_memory, recall, search) for Graph-Memory-for-Local-LLM
]


def execute_tool(name, args):
    if name == "read_file":
        try:
            with open(args["path"], "r") as f:
                lines = f.readlines()
                start = args.get("offset", 1) - 1
                end = start + args.get("limit", 2000)
                return "".join(lines[start:end])
        except Exception as e:
            return f"Error: {str(e)}"
    elif name == "write_file":
        try:
            with open(args["path"], "w") as f:
                f.write(args["content"])
            return f"Successfully wrote to {args['path']}"
        except Exception as e:
            return f"Error: {str(e)}"
    elif name == "edit_file":
        try:
            with open(args["path"], "r") as f:
                content = f.read()
            if args["old_string"] not in content:
                return f"Error: Old string not found in {args['path']}"
            new_content = content.replace(
                args["old_string"], args["new_string"], 1
            )  # Replace first occurrence
            with open(args["path"], "w") as f:
                f.write(new_content)
            return f"Successfully edited {args['path']}"
        except Exception as e:
            return f"Error: {str(e)}"
    elif name == "file_glob":
        try:
            pattern = args["pattern"]
            root_dir = args.get("root_dir", ".")
            # Change to root_dir temporarily for glob
            old_cwd = os.getcwd()
            os.chdir(root_dir)
            matches = glob.glob(pattern, recursive=True)
            os.chdir(old_cwd)
            return json.dumps(matches)
        except Exception as e:
            return f"Error: {str(e)}"
    elif name == "run_shell_command":
        try:
            result = subprocess.run(
                args["command"], shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Error: {str(e)}"
    elif name == "web_fetch":
        try:
            req = request.Request(
                args["url"], headers={"User-Agent": "Mozilla/5.0 (Qwen Local Agent)"}
            )
            with request.urlopen(req, timeout=args.get("timeout", 10)) as response:
                content = response.read().decode("utf-8", errors="ignore")
                # Limit content length to prevent overwhelming the context
                max_len = 8000
                if len(content) > max_len:
                    content = content[:max_len] + "\n... [content truncated]"
                return content
        except error.URLError as e:
            return f"Error fetching URL: {str(e.reason)}"
        except Exception as e:
            return f"Error: {str(e)}"
    return "Unknown tool"


def parse_tool_from_content(content):
    content = content.strip()

    # Handle direct JSON
    try:
        tool_data = json.loads(content)
        if "name" in tool_data and "arguments" in tool_data:
            return tool_data
    except json.JSONDecodeError:
        pass

    # Handle JSON string in content like {"name": "read_file", "arguments": {...}}
    try:
        tool_data = json.loads(content)
        if "name" in tool_data and "arguments" in tool_data:
            return tool_data
    except json.JSONDecodeError:
        pass

    # Handle JSON in markdown code block like ```json {...} ```
    import re

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        try:
            tool_data = json.loads(json_match.group(1))
            if "name" in tool_data and "arguments" in tool_data:
                return tool_data
        except json.JSONDecodeError:
            pass

    # Handle XML format like <tools>{"name": "...", "arguments": {...}}</tools>
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(content)
        # Look for tools tag first
        tools_tag = root.find("tools")
        if tools_tag is not None and tools_tag.text:
            # Parse the JSON inside the tools tag
            tool_data = json.loads(tools_tag.text.strip())
            if "name" in tool_data and "arguments" in tool_data:
                return tool_data
        # Also check for direct function tag
        func = root.find("function")
        if func is not None:
            name = func.get("name")
            args_str = func.get("arguments", "{}")
            args = json.loads(args_str)
            return {"name": name, "arguments": args}
    except:
        pass
    return None


def chat_with_qwen(prompt):
    messages = [{"role": "user", "content": prompt}]
    max_iterations = 10  # Increased for complex chains
    for i in range(max_iterations):
        payload = {
            "model": "qwen3-coder",
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
        except json.JSONDecodeError as e:
            return f"JSON Decode Error: {str(e)}"

        if not data.get("choices"):
            return "Invalid response: no choices"

        msg = data["choices"][0]["message"]
        messages.append(msg)
        content = msg.get("content", "")
        tool_call = parse_tool_from_content(content)
        if tool_call:
            result = execute_tool(tool_call["name"], tool_call["arguments"])
            # Format result for better LLM consumption
            if isinstance(result, str) and len(result) > 100:
                # Truncate very long results but keep them useful
                display_result = result[:200] + ("..." if len(result) > 200 else "")
                messages.append(
                    {"role": "tool", "content": f"Result: {display_result}"}
                )
            else:
                messages.append({"role": "tool", "content": f"Result: {result}"})
        else:
            return content  # Final response
    return "Max iterations reached - possible infinite loop"


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, read the README.md file"
    print(chat_with_qwen(prompt))
