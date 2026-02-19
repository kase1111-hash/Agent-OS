"""
Tool Subprocess Runner

Standalone entry point for executing tools in isolated subprocess environments.
This module is invoked by SandboxedExecutor._execute_subprocess() to provide
actual process isolation for tool execution.

Usage:
    python -m src.tools.subprocess_runner \
        --tool-module src.tools.example \
        --tool-class ExampleTool \
        --tool-name example \
        --params-file /tmp/params.json

Security notes:
- This process runs with a restricted environment (sensitive env vars stripped)
- Parameters are read from a file (not command-line args) to avoid shell injection
- Output is JSON on stdout; errors go to stderr
"""

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Execute a tool in subprocess isolation")
    parser.add_argument("--tool-module", required=True, help="Python module path")
    parser.add_argument("--tool-class", required=True, help="Tool class name")
    parser.add_argument("--tool-name", required=True, help="Tool name")
    parser.add_argument("--params-file", required=True, help="Path to parameters JSON file")

    args = parser.parse_args()

    try:
        # Read parameters from file (not CLI args)
        params_path = Path(args.params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {args.params_file}")

        with open(params_path) as f:
            parameters = json.load(f)

        # Import and instantiate the tool
        module = importlib.import_module(args.tool_module)
        tool_class = getattr(module, args.tool_class)
        try:
            tool_instance = tool_class(args.tool_name)
        except TypeError:
            # Some tools don't accept a name parameter
            tool_instance = tool_class()

        # Execute
        start_time = time.time()
        result = tool_instance.invoke(parameters)
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Output result as JSON
        output = {
            "success": True,
            "result": result.result.name if hasattr(result.result, "name") else str(result.result),
            "output": result.output,
            "error": result.error,
            "execution_time_ms": execution_time_ms,
        }
        print(json.dumps(output))

    except Exception as e:
        output = {
            "success": False,
            "error": type(e).__name__,
            "execution_time_ms": 0,
        }
        print(json.dumps(output))
        logger.error(f"Subprocess runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
