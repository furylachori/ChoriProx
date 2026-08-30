#!/usr/bin/env python3
"""
Enhanced E2E Test Prompts with Improved Validation
Addresses adversary review findings
"""

from typing import Dict, List, Any, Optional
from enhanced_test_infrastructure import EnhancedTestValidator


class EnhancedPrompts:
    """Enhanced prompts with probabilistic validation"""

    def __init__(self, test_data_dir: str = "/tmp/qwen_test_data"):
        self.validator = EnhancedTestValidator(test_data_dir)
        self.test_matrix = self.validator.generate_test_matrix()

    def get_enhanced_prompts(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Get enhanced prompts with validation criteria"""

        return {
            "batch1_enhanced": {
                "tier1_file_operations": [
                    {
                        "prompt": f"Read the file {self.test_matrix['syntax_errors'][0]} and identify any syntax errors",
                        "input_files": [self.test_matrix["syntax_errors"][0]],
                        "validation_type": "error_detection",
                        "expected_patterns": ["missing colon", "SyntaxError"],
                        "probabilistic_threshold": 0.7,
                    },
                    {
                        "prompt": f"Write a new file /tmp/qwen_test_data/output/processed_data.json with structured analysis of {self.test_matrix['csv_processing'][0]}",
                        "input_files": [self.test_matrix["csv_processing"][0]],
                        "validation_type": "data_processing",
                        "expected_patterns": ["data", "columns", "row_count"],
                        "probabilistic_threshold": 0.8,
                    },
                ],
                "tier1_search_patterns": [
                    {
                        "prompt": "Find all Python files in /tmp/qwen_test_data/input_files/python that contain syntax errors",
                        "input_files": list(self.test_matrix["syntax_errors"]),
                        "validation_type": "pattern_matching",
                        "expected_patterns": [".py", "syntax_error"],
                        "probabilistic_threshold": 0.9,
                    },
                    {
                        "prompt": "Search for files containing Unicode characters in /tmp/qwen_test_data/input_files/edge_cases",
                        "input_files": [self.test_matrix["edge_cases"][0]],
                        "validation_type": "unicode_detection",
                        "expected_patterns": ["unicode", "José", "São"],
                        "probabilistic_threshold": 0.6,
                    },
                ],
                "tier1_system_commands": [
                    {
                        "prompt": "List all files in /tmp/qwen_test_data/input_files and count them by type",
                        "input_files": [],
                        "validation_type": "file_listing",
                        "expected_patterns": [".py", ".json", ".csv", "count"],
                        "probabilistic_threshold": 0.8,
                    }
                ],
            },
            "batch2_enhanced": {
                "tier1_code_generation_chain": [
                    {
                        "prompt": f'Create CSV processing pipeline with defined contracts for {self.test_matrix["csv_processing"][1]}:\n1. CSV Parser: Extract data from CSV file\n   - Input: {self.test_matrix["csv_processing"][1]}\n   - Output: JSON at /tmp/qwen_test_data/output/parser_output.json with schema {{"headers": ["col1", "col2", ...], "rows": [["val1", "val2", ...]], "row_count": N}}\n2. Data Transformer: Normalize column names and data types\n   - Input: Parser JSON output\n   - Output: JSON at /tmp/qwen_test_data/output/transformer_output.json with schema {{"cleaned_data": [...], "normalized_fields": {{...}}, "transformation_stats": {{...}}}}\n3. Report Generator: Create analysis summary\n   - Input: Transformer JSON output\n   - Output: JSON at /tmp/qwen_test_data/output/pipeline_report.json with schema {{"statistics": {{...}}, "validation_results": {{...}}, "summary": {{...}}}}\n4. Validation: Run /tmp/qwen_test_data/scripts/validate_pipeline.py\n   - Test passes if script returns exit code 0 and outputs \'VALIDATION_PASSED\'',
                        "input_files": [self.test_matrix["csv_processing"][1]],
                        "validation_type": "tool_chain",
                        "contract_schemas": [
                            "csv_parser_output",
                            "data_transformer_output",
                            "report_generator_output",
                        ],
                        "probabilistic_threshold": 0.9,
                    }
                ],
                "tier2_multi_tool_code_chain": [
                    {
                        "prompt": f'Build a 3-stage error analysis pipeline for {self.test_matrix["syntax_errors"][1]}:\n1. Code Analyzer: Parse the file and identify syntax errors\n   - Input: {self.test_matrix["syntax_errors"][1]}\n   - Output: JSON with schema {{"syntax_errors": [{{"line": N, "column": N, "error_type": "text", "description": "text"}}], "file_analysis": {{"total_lines": N, "function_count": N}}}}\n2. Fix Generator: Apply fixes based on analyzer output\n   - Input: Analyzer JSON output\n   - Output: Corrected Python file at /tmp/qwen_test_data/output/fixed_code.py\n3. Validator: Execute /tmp/qwen_test_data/scripts/validate_fix.py\n   - Test passes if script returns exit code 0 and outputs \'VALIDATION_PASSED\'',
                        "input_files": [self.test_matrix["syntax_errors"][1]],
                        "validation_type": "error_analysis_chain",
                        "expected_patterns": [
                            "VALIDATION_PASSED",
                            "fixed_code.py",
                            "syntax_errors",
                        ],
                        "probabilistic_threshold": 0.9,
                    }
                ],
                "tier3_bug_fix_flow": [
                    {
                        "prompt": f"Analyze and fix the Python file {self.test_matrix['syntax_errors'][2]}:\n1) Find all syntax errors\n2) Generate corrected code\n3) Verify the fix by checking if it would run",
                        "input_files": [self.test_matrix["syntax_errors"][2]],
                        "validation_type": "bug_fix_validation",
                        "expected_patterns": [
                            "missing colon",
                            "undefined variable",
                            "def.*:",
                            "class.*:",
                        ],
                        "probabilistic_threshold": 0.7,
                    }
                ],
            },
            "batch3_enhanced": {
                "memory_pressure": [
                    {
                        "prompt": f'Process all files in /tmp/qwen_test_data/input_files and generate structured analysis report:\n1. For each file type (.py, .json, .csv):\n   - Count files\n   - Extract metadata (size, line count, encoding)\n   - Identify file categories (syntax errors, valid, invalid)\n2. Save analysis report to /tmp/qwen_test_data/output/analysis_report.json with schema:\n   {{"file_types": {{"python": count, "json": count, "csv": count}}, "total_files": N, "analysis_summary": "text"}}\n3. Run validation script: /tmp/qwen_test_data/scripts/validate_analysis.py\n4. The test passes if validation script returns exit code 0 and outputs \'VALIDATION_PASSED\'',
                        "input_files": [
                            f for sublist in self.test_matrix.values() for f in sublist
                        ],
                        "validation_type": "large_scale_processing",
                        "expected_patterns": [
                            "VALIDATION_PASSED",
                            "analysis_report.json",
                            "file_types",
                        ],
                        "probabilistic_threshold": 0.9,
                    }
                ],
                "context_overflow": [
                    {
                        "prompt": "Execute comprehensive multi-stage workflow:\n1) Setup test environment\n2) Analyze all input files\n3) Generate transformation pipelines\n4) Create validation reports\n5) Cleanup temporary files",
                        "input_files": [
                            f for sublist in self.test_matrix.values() for f in sublist
                        ],
                        "validation_type": "complex_workflow",
                        "expected_patterns": [
                            "setup",
                            "analysis",
                            "pipeline",
                            "report",
                            "cleanup",
                        ],
                        "probabilistic_threshold": 0.6,
                    }
                ],
            },
            "batch4_enhanced": {
                "performance_validation": [
                    {
                        "prompt": f"Process {self.test_matrix['csv_processing'][2]} and measure execution time for:\n1) File reading\n2) Data parsing\n3) Report generation",
                        "input_files": [self.test_matrix["csv_processing"][2]],
                        "validation_type": "timing_benchmark",
                        "expected_patterns": ["time", "execution", "performance"],
                        "probabilistic_threshold": 0.8,
                    }
                ],
                "cache_validation": [
                    {
                        "prompt": f"Execute the same operation twice on {self.test_matrix['json_validation'][0]} and compare results for consistency",
                        "input_files": [self.test_matrix["json_validation"][0]],
                        "validation_type": "cache_consistency",
                        "expected_patterns": ["same result", "consistent", "cached"],
                        "probabilistic_threshold": 0.9,
                    }
                ],
            },
        }

    def get_prompt_with_validation(
        self, batch: str, test_name: str, variant: int = 0
    ) -> Dict[str, Any]:
        """Get enhanced prompt with validation metadata"""
        all_prompts = self.get_enhanced_prompts()

        if batch in all_prompts and test_name in all_prompts[batch]:
            prompts = all_prompts[batch][test_name]
            if variant < len(prompts):
                return prompts[variant]

        return {
            "prompt": f"Enhanced test for {batch}.{test_name}",
            "input_files": [],
            "validation_type": "generic",
            "expected_patterns": [],
            "probabilistic_threshold": 0.5,
        }

    def validate_test_result(
        self,
        prompt_metadata: Dict[str, Any],
        actual_output: str,
        intermediate_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Validate test result using enhanced validation"""
        validation_type = prompt_metadata.get("validation_type", "generic")
        expected_patterns = prompt_metadata.get("expected_patterns", [])
        threshold = prompt_metadata.get("probabilistic_threshold", 0.5)

        results = {
            "valid": False,
            "validation_type": validation_type,
            "pattern_matches": [],
            "similarity_score": 0.0,
            "contract_valid": False,
            "issues": [],
        }

        # Pattern-based validation
        for pattern in expected_patterns:
            if self.validator.probabilistic_validator.validate_with_patterns(
                actual_output, "generic", threshold=0.3
            ):
                results["pattern_matches"].append(pattern)

        # Contract validation for tool chains
        if validation_type == "tool_chain" and intermediate_data:
            contract_schemas = prompt_metadata.get("contract_schemas", [])
            if contract_schemas:
                results["contract_valid"] = (
                    self.validator.contract_validator.validate_schema(
                        intermediate_data, contract_schemas[0]
                    )
                )

        # Error analysis validation
        if validation_type == "error_analysis_chain":
            input_files = prompt_metadata.get("input_files", [])
            if input_files:
                error_results = self.validator.validate_error_analysis(
                    input_files[0], actual_output, expected_patterns
                )
                results.update(error_results)

        # Determine overall validity
        pattern_match_ratio = len(results["pattern_matches"]) / max(
            1, len(expected_patterns)
        )
        results["valid"] = (
            pattern_match_ratio >= threshold
            and (not results.get("contract_valid", True) or results["contract_valid"])
            and (
                not results.get("similarity_score", 1.0)
                or results["similarity_score"] > 0.1
            )
        )

        return results


# Usage example
if __name__ == "__main__":
    enhanced_prompts = EnhancedPrompts()

    print("🚀 Enhanced E2E Test Prompts")
    print("=" * 50)

    all_enhanced_prompts = enhanced_prompts.get_enhanced_prompts()

    total_tests = 0
    for batch_name, batch_tests in all_enhanced_prompts.items():
        print(f"\n{batch_name.upper()}:")
        for test_name, tests in batch_tests.items():
            print(f"  {test_name}: {len(tests)} enhanced prompts")
            total_tests += len(tests)

    print(f"\n📊 Total Enhanced Prompts: {total_tests}")

    # Example validation
    sample_prompt = enhanced_prompts.get_prompt_with_validation(
        "batch1_enhanced", "tier1_file_operations", 0
    )
    print(f"\n📝 Sample Enhanced Prompt:")
    print(f"  Prompt: {sample_prompt['prompt']}")
    print(f"  Input Files: {sample_prompt['input_files']}")
    print(f"  Validation Type: {sample_prompt['validation_type']}")
    print(f"  Expected Patterns: {sample_prompt['expected_patterns']}")

    print("\n✅ Enhanced prompt system ready with probabilistic validation!")
