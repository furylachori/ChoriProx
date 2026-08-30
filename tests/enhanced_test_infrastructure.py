#!/usr/bin/env python3
"""
Enhanced E2E Test Infrastructure
Addresses adversary review findings with improved validation
"""

import os
import re
import json
import difflib
import random
import string
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path


class ProbabilisticValidator:
    """Probabilistic validation instead of exact matches"""

    def __init__(self):
        self.patterns = self._load_validation_patterns()

    def _load_validation_patterns(self):
        """Load regex patterns for validation"""
        return {
            "syntax_error": [
                r"SyntaxError.*[:\n]",
                r"missing colon",
                r"undefined.*variable",
                r"invalid syntax",
            ],
            "json_validation": [
                r"JSON.*valid",
                r"parsing.*success",
                r"invalid.*JSON",
                r"expecting.*property",
            ],
            "code_fix": [
                r"def\s+\w+\(.*\):",  # Function definition
                r"return\s+",  # Return statement
                r"import\s+",  # Import statement
                r"class\s+\w+",  # Class definition
            ],
        }

    def validate_with_patterns(
        self, output: str, pattern_type: str, threshold: float = 0.7
    ) -> bool:
        """Validate output using regex patterns"""
        if pattern_type not in self.patterns:
            return False

        patterns = self.patterns[pattern_type]
        matches = 0

        for pattern in patterns:
            if re.search(pattern, output, re.IGNORECASE):
                matches += 1

        return (matches / len(patterns)) >= threshold

    def semantic_similarity(
        self, output1: str, output2: str, threshold: float = 0.8
    ) -> bool:
        """Check semantic similarity between outputs"""
        # Simple similarity check - can be enhanced with NLP
        similarity = difflib.SequenceMatcher(
            None, output1.lower(), output2.lower()
        ).ratio()
        return similarity >= threshold


class ContractValidator:
    """Validate input/output contracts between chained tools"""

    def __init__(self, test_data_dir: str):
        self.test_data_dir = test_data_dir
        self.schemas = self._load_schemas()

    def _load_schemas(self):
        """Load JSON schemas for contract validation"""
        return {
            "csv_parser_output": {
                "type": "object",
                "required": ["headers", "rows", "row_count"],
                "properties": {
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "row_count": {"type": "integer"},
                },
            },
            "data_transformer_output": {
                "type": "object",
                "required": [
                    "cleaned_data",
                    "normalized_fields",
                    "transformation_stats",
                ],
                "properties": {
                    "cleaned_data": {"type": "array"},
                    "normalized_fields": {"type": "object"},
                    "transformation_stats": {"type": "object"},
                },
            },
            "report_generator_output": {
                "type": "object",
                "required": ["statistics", "validation_results", "summary"],
                "properties": {
                    "statistics": {"type": "object"},
                    "validation_results": {"type": "object"},
                    "summary": {"type": "object"},
                },
            },
            "code_analyzer_output": {
                "type": "object",
                "required": ["syntax_errors", "file_analysis"],
                "properties": {
                    "syntax_errors": {"type": "array"},
                    "file_analysis": {"type": "object"},
                },
            },
            "fix_generator_output": {
                "type": "object",
                "required": ["fixed_code", "applied_fixes"],
                "properties": {
                    "fixed_code": {"type": "string"},
                    "applied_fixes": {"type": "array"},
                },
            },
            "code_fix_output": {
                "type": "object",
                "required": ["fixed_code", "changes"],
                "properties": {
                    "fixed_code": {"type": "string"},
                    "changes": {"type": "array"},
                },
            },
        }

    def validate_schema(self, data: Any, schema_name: str) -> bool:
        """Validate data against schema"""
        if schema_name not in self.schemas:
            return False

        schema = self.schemas[schema_name]

        try:
            # Basic schema validation
            if not isinstance(data, dict):
                return False

            # Check required fields
            for field in schema.get("required", []):
                if field not in data:
                    return False

            # Check field types
            for field, field_schema in schema.get("properties", {}).items():
                if field in data:
                    expected_type = field_schema.get("type")
                    if expected_type and not isinstance(
                        data[field], self._get_type(expected_type)
                    ):
                        return False

            return True
        except:
            return False

    def _get_type(self, type_str: str) -> type:
        """Convert string type to Python type"""
        type_map = {
            "string": str,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(type_str, object)

    def check_tool_chain_compatibility(
        self, output_schema: str, input_schema: str
    ) -> bool:
        """Check if two tools can be chained"""
        # This would check if output schema matches expected input schema
        # For now, return True if both schemas exist
        return output_schema in self.schemas and input_schema in self.schemas


class TestDataGenerator:
    """Generate realistic test data with parameterization"""

    def __init__(self, test_data_dir: str):
        self.test_data_dir = test_data_dir
        self.setup_directories()

    def setup_directories(self):
        """Create directory structure for enhanced testing"""
        directories = [
            "python/syntax_errors",
            "python/type_errors",
            "python/import_errors",
            "python/large_codebases",
            "json/valid",
            "json/invalid",
            "json/schema_violations",
            "csv/clean",
            "csv/messy",
            "csv/encoding_issues",
            "edge_cases/binary_data",
            "edge_cases/circular_references",
            "edge_cases/corrupted_headers",
            "edge_cases/permissions",
            "edge_cases/unicode",
            "edge_cases/large_file",
        ]

        for directory in directories:
            Path(f"{self.test_data_dir}/input_files/{directory}").mkdir(
                parents=True, exist_ok=True
            )

    def generate_syntax_error_file(self, complexity: str = "medium") -> str:
        """Generate Python file with syntax errors"""
        errors = {
            "simple": ["def test_function()\n    return True"],  # Missing colon
            "medium": [
                "def test_function()\n    return True",
                "undefined_var = unknown_var + 1",  # Undefined variable
            ],
            "complex": [
                "def test_function()\n    return True",
                "undefined_var = unknown_var + 1",
                "class TestClass\n    def __init__(self):\n        pass",  # Missing colon
            ],
        }

        code_lines = errors.get(complexity, errors["medium"])
        file_path = f"{self.test_data_dir}/input_files/python/syntax_errors/syntax_error_{complexity}.py"

        with open(file_path, "w") as f:
            f.write("\n".join(code_lines))

        return file_path

    def generate_messy_csv(self, rows: int = 10, encoding: str = "utf-8") -> str:
        """Generate messy CSV data"""
        file_path = f"{self.test_data_dir}/input_files/csv/messy/data_{rows}_rows.csv"

        headers = ["Name", "Age", "Email", "Salary"]
        data = []

        for i in range(rows):
            row = [
                f"User {i}",
                str(random.randint(18, 65)),
                f"user{i}@example.com",
                f"${random.randint(30000, 120000)}",
            ]

            # Add some messiness
            if i % 3 == 0:
                row[1] = ""  # Missing age
            if i % 5 == 0:
                row[3] = "invalid_salary"  # Invalid data

            data.append(row)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(",".join(headers) + "\n")
            for row in data:
                f.write(",".join(row) + "\n")

        return file_path

    def generate_invalid_json(self, error_type: str = "missing_comma") -> str:
        """Generate invalid JSON files"""
        errors = {
            "missing_comma": '{"name": "John" "age": 30}',
            "unclosed_brace": '{"name": "John", "age": 30',
            "invalid_value": '{"name": "John", "age": "thirty"}',
            "trailing_comma": '{"name": "John", "age": 30,}',
            "unquoted_key": '{name: "John", "age": 30}',
        }

        json_content = errors.get(error_type, errors["missing_comma"])
        file_path = f"{self.test_data_dir}/input_files/json/invalid/{error_type}.json"

        with open(file_path, "w") as f:
            f.write(json_content)

        return file_path

    def generate_edge_case_file(self, edge_case_type: str) -> Optional[str]:
        """Generate edge case files"""
        edge_cases = {
            "unicode": {
                "content": "# File with Unicode characters\nname = 'José Müller'\ncity = 'São Paulo'\nemoji = '🚀 Python 🐍'",
                "extension": "py",
            },
            "binary_data": {
                "content": b"Binary data test\x00\x01\x02\x03",
                "extension": "bin",
            },
            "large_file": {
                "content": "\n".join(
                    [f"Line {i}: This is a large file for testing" for i in range(1000)]
                ),
                "extension": "txt",
            },
        }

        if edge_case_type not in edge_cases:
            return None

        case = edge_cases[edge_case_type]
        file_path = f"{self.test_data_dir}/input_files/edge_cases/{edge_case_type}/test.{case['extension']}"

        mode = "wb" if isinstance(case["content"], bytes) else "w"
        with open(file_path, mode) as f:
            f.write(case["content"])

        return file_path


class EnhancedTestValidator:
    """Main enhanced validation class"""

    def __init__(self, test_data_dir: str = "/tmp/qwen_test_data"):
        self.test_data_dir = test_data_dir
        self.probabilistic_validator = ProbabilisticValidator()
        self.contract_validator = ContractValidator(test_data_dir)
        self.data_generator = TestDataGenerator(test_data_dir)

    def validate_error_analysis(
        self, original_file: str, fixed_output: str, expected_patterns: List[str]
    ) -> Dict[str, Any]:
        """Validate error analysis with probabilistic patterns"""
        results = {
            "valid": False,
            "patterns_matched": [],
            "similarity_score": 0.0,
            "issues_found": [],
        }

        # Check if output contains expected patterns
        for pattern in expected_patterns:
            if re.search(pattern, fixed_output, re.IGNORECASE):
                results["patterns_matched"].append(pattern)

        # Read original file for comparison
        try:
            with open(original_file, "r") as f:
                original_content = f.read()

            # Basic similarity check
            similarity = difflib.SequenceMatcher(
                None, original_content, fixed_output
            ).ratio()
            results["similarity_score"] = similarity

            # Check if syntax errors were likely fixed
            syntax_errors_fixed = self.probabilistic_validator.validate_with_patterns(
                fixed_output, "code_fix", threshold=0.5
            )

            results["valid"] = (
                len(results["patterns_matched"]) >= len(expected_patterns) * 0.6
                and syntax_errors_fixed
                and similarity < 0.95  # Should be different from original
            )

        except Exception as e:
            results["issues_found"].append(f"Error reading original file: {e}")

        return results

    def validate_data_processing_chain(
        self, input_file: str, intermediate_output: Any, final_output: str
    ) -> Dict[str, Any]:
        """Validate data processing tool chain"""
        results = {
            "valid": False,
            "contract_checks": {},
            "format_validation": {},
            "data_integrity": {},
        }

        # Check intermediate output contract
        if isinstance(intermediate_output, dict):
            results["contract_checks"]["intermediate"] = (
                self.contract_validator.validate_schema(
                    intermediate_output, "csv_parser_output"
                )
            )

        # Check final output format
        results["format_validation"]["final_output"] = bool(final_output.strip())

        # Basic data integrity checks
        if "data" in intermediate_output:
            results["data_integrity"]["row_count"] = (
                len(intermediate_output["data"]) > 0
            )

        results["valid"] = (
            results["contract_checks"].get("intermediate", False)
            and results["format_validation"].get("final_output", False)
            and results["data_integrity"].get("row_count", False)
        )

        return results

    def generate_test_matrix(self) -> Dict[str, List[str]]:
        """Generate systematic test matrix"""
        matrix = {
            "syntax_errors": [
                self.data_generator.generate_syntax_error_file("simple"),
                self.data_generator.generate_syntax_error_file("medium"),
                self.data_generator.generate_syntax_error_file("complex"),
            ],
            "json_validation": [
                self.data_generator.generate_invalid_json("missing_comma"),
                self.data_generator.generate_invalid_json("unclosed_brace"),
                self.data_generator.generate_invalid_json("invalid_value"),
            ],
            "csv_processing": [
                self.data_generator.generate_messy_csv(5),
                self.data_generator.generate_messy_csv(20),
                self.data_generator.generate_messy_csv(50),
            ],
            "edge_cases": [
                self.data_generator.generate_edge_case_file("unicode"),
                self.data_generator.generate_edge_case_file("large_file"),
            ],
        }

        return matrix


# Usage example
if __name__ == "__main__":
    validator = EnhancedTestValidator()

    # Generate test matrix
    test_matrix = validator.generate_test_matrix()
    print("📊 Generated Test Matrix:")
    for category, files in test_matrix.items():
        print(f"  {category}: {len(files)} files")

    print("\n✅ Enhanced test infrastructure ready!")
