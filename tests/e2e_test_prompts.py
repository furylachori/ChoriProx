#!/usr/bin/env python3
"""
E2E Test Prompts for Tool Caching Pipeline
Contains all 27 specific prompts for the test scenarios
"""

# BATCH 1: INDEPENDENT TOOLS PROMPTS
BATCH1_PROMPTS = {
    "tier1_file_operations": [
        "Read the file /tmp/qwen_test_data/simple.txt and tell me its contents",
        "Write a new file /tmp/qwen_test_data/output.txt with the content 'Test output data for file operations'",
        "Edit the file /tmp/qwen_test_data/simple.txt to add a timestamp at the end",
    ],
    "tier1_search_patterns": [
        "Find all .txt files in /tmp/qwen_test_data",
        "Search for the word 'test' in all files in /tmp/qwen_test_data",
        "Find Python files in the current directory that contain function definitions",
    ],
    "tier1_system_commands": [
        "List files in the current directory",
        "Check the git status of this repository",
        "Show the current working directory",
    ],
    "tier2_web_operations": [
        "Fetch the content from http://httpbin.org/json",
        "Get information from http://httpbin.org/headers",
        "Download and analyze the structure of http://httpbin.org/html",
    ],
    "tier2_code_generation_chains": [
        "Write a Python function that parses JSON data and returns a structured dict with 'name', 'age', 'email' fields",
        "Write a Python function that validates the output from the JSON parser and returns success/failure status",
        "Write a Python function that formats the validated data into a human-readable report",
    ],
    "tier2_complex_tasks": [
        "Create a comprehensive test suite for a data processing module",
        "Analyze the project structure and suggest architectural improvements",
        "Design a caching strategy for a web API with multiple endpoints",
    ],
    "tier3_error_scenarios": [
        "Try to read the file /nonexistent/path/file.txt",
        "Execute the command 'invalid_command_xyz' in bash",
        "Search for files in /invalid/directory/path",
    ],
    "tier3_large_data": [
        "Read the large file /tmp/qwen_test_data/large_data.txt and count the lines",
        "Search for patterns in /tmp/qwen_test_data/large_data.txt and extract statistics",
        "Process all large files in /tmp/qwen_test_data and generate a summary",
    ],
    "tier3_mixed_tool_types": [
        "Find Python files in this directory, read one, and search for function definitions",
        "Fetch web documentation about Python, extract key concepts, and write a summary file",
        "Analyze the project structure, find configuration files, and suggest improvements",
    ],
}

# BATCH 2: CHAINED TOOLS PROMPTS
BATCH2_PROMPTS = {
    "tier1_file_pipeline": [
        "Find text files in /tmp/qwen_test_data, read one, add a timestamp, and save it",
        "Search for JSON files, read their contents, validate the structure, and write a validation report",
        "Find Python files, extract function names, create a documentation file with function signatures",
    ],
    "tier1_web_research": [
        "Fetch Python documentation about lists, extract key information, and write a summary",
        "Get API documentation from http://httpbin.org, analyze the endpoints, and create a usage guide",
        "Research best practices for Python testing, extract key points, and write a testing guide",
    ],
    "tier1_code_generation_chain": [
        "Create a 3-tool data processing pipeline: 1) JSON parser, 2) data transformer, 3) report formatter",
        "Build a 3-tool validation system: 1) input validator, 2) data processor, 3) output formatter",
        "Develop a 3-tool analysis toolkit: 1) data loader, 2) statistical analyzer, 3) visualization generator",
    ],
    "tier2_code_analysis": [
        "Analyze Python files in this project, find function definitions, and suggest refactoring improvements",
        "Examine the test files, identify test patterns, and recommend testing strategy enhancements",
        "Review the project structure, find dependencies between modules, and suggest architectural changes",
    ],
    "tier2_multi_tool_code_chain": [
        "Create a 3-tool data processing system: 1) CSV data extractor, 2) data transformer for normalization, 3) data validator with schema checking",
        "Build a 3-tool API client: 1) request builder, 2) response parser, 3) error handler",
        "Develop a 3-tool logging system: 1) log formatter, 2) log writer, 3) log analyzer",
    ],
    "tier2_multi_step_setup": [
        "Set up a Python project: create requirements.txt, write a main.py file, install dependencies, and test the setup",
        "Configure a testing environment: create test files, set up test configuration, run initial tests, and report results",
        "Initialize a documentation project: create docs structure, write initial content, build documentation, and verify output",
    ],
    "tier3_bug_fix_flow": [
        "Find errors in Python files, examine the problematic code, fix the issues, and verify the fixes work",
        "Identify performance bottlenecks in the code, analyze the root causes, optimize the code, and measure improvements",
        "Detect security vulnerabilities, examine the vulnerable code, implement security fixes, and validate the changes",
    ],
    "tier3_feature_development": [
        "Develop a new feature: write specifications, implement the code, get code review, and make improvements",
        "Create a new module: design the API, implement the functionality, write tests, and document usage",
        "Build a new integration: define requirements, implement connectors, test integration, and optimize performance",
    ],
    "tier3_documentation_update": [
        "Update project documentation: research best practices, read current docs, make improvements, and write final version",
        "Modernize API documentation: analyze current docs, research modern standards, rewrite documentation, and validate clarity",
        "Enhance user guides: gather user feedback, analyze existing guides, improve content, and test usability",
    ],
}

# BATCH 3: STRESS & FAILURE PROMPTS
BATCH3_PROMPTS = {
    "memory_pressure": [
        "Read and process all large files in /tmp/qwen_test_data",
        "Search through all files in the project for multiple patterns simultaneously",
        "Execute multiple system commands and process their outputs",
    ],
    "context_overflow": [
        "Perform a comprehensive analysis: find all files, read their contents, search for patterns, analyze structure, and generate detailed reports for each file type",
        "Execute a multi-stage workflow: setup environment, configure components, run tests, analyze results, generate reports, and cleanup",
        "Conduct a full project audit: examine code quality, test coverage, documentation, performance, security, and architecture",
    ],
    "tool_errors": [
        "Try to read non-existent files and execute invalid commands",
        "Attempt operations with malformed inputs and incorrect parameters",
        "Test edge cases with boundary conditions and unexpected inputs",
    ],
    "concurrent_access": [
        "List files in current directory",
        "Check system information",
        "Read configuration files",
    ],
}

# BATCH 4: PERFORMANCE VALIDATION PROMPTS
BATCH4_PROMPTS = {
    "timing_benchmarks": [
        "Perform multiple file operations and system commands",
        "Execute a series of search and analysis operations",
        "Run comprehensive code analysis and generation tasks",
    ],
    "cache_hit_validation": [
        "List files in current directory",
        "Check git status",
        "Read project configuration",
    ],
}

# COMPLETE TEST SUITE
ALL_PROMPTS = {
    "batch1": BATCH1_PROMPTS,
    "batch2": BATCH2_PROMPTS,
    "batch3": BATCH3_PROMPTS,
    "batch4": BATCH4_PROMPTS,
}


def get_prompt(test_category: str, test_name: str, variant: int = 0) -> str:
    """Get a specific prompt for testing."""
    category_prompts = ALL_PROMPTS.get(test_category, {})
    test_prompts = category_prompts.get(test_name, [])

    if test_prompts and variant < len(test_prompts):
        return test_prompts[variant]
    elif test_prompts:
        return test_prompts[0]
    else:
        return f"Test prompt for {test_category}.{test_name}"


def get_all_prompts() -> dict:
    """Get all prompts organized by batch and test."""
    return ALL_PROMPTS


if __name__ == "__main__":
    # Print all prompts for verification
    print("📋 E2E TEST PROMPTS SUMMARY")
    print("=" * 50)

    total_prompts = 0
    for batch_name, batch_prompts in ALL_PROMPTS.items():
        print(f"\n{batch_name.upper()}:")
        for test_name, prompts in batch_prompts.items():
            print(f"  {test_name}: {len(prompts)} prompts")
            total_prompts += len(prompts)

    print(f"\n📊 Total Prompts: {total_prompts}")
    print("✅ Prompt library ready for E2E testing")
