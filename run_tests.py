#!/usr/bin/env python3
"""
Test runner script for chatbot boilerplate.
This script provides convenient commands to run different test suites.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest pytest-asyncio")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for chatbot boilerplate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --coverage         # Run tests with coverage report
  python run_tests.py --verbose          # Run tests with verbose output
  python run_tests.py --fast             # Run tests with minimal output
        """
    )
    
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run only unit tests (utils, states, nodes)"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run only integration tests (agent, graph)"
    )
    parser.add_argument(
        "--edge-cases", 
        action="store_true", 
        help="Run only edge case tests"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Run tests with verbose output"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run tests with minimal output"
    )
    parser.add_argument(
        "--parallel", "-n", 
        type=int, 
        help="Run tests in parallel (requires pytest-xdist)"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    # Determine test selection
    if args.unit:
        test_files = ["tests/test_chat_utils.py", "tests/test_chat_state.py", "tests/test_chat_nodes.py"]
    elif args.integration:
        test_files = ["tests/test_chat_agent.py", "tests/test_chat_graph.py"]
    elif args.edge_cases:
        test_files = ["tests/test_edge_cases.py"]
    elif args.file:
        test_files = [args.file]
    else:
        test_files = ["tests/"]
    
    # Build command
    cmd = base_cmd + test_files
    
    # Add options
    if args.coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing", "--cov-report=html"])
    
    if args.verbose:
        cmd.append("-v")
    elif args.fast:
        cmd.append("-q")
    
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Always show local variables on failure for better debugging
    cmd.append("--tb=short")
    
    # Run the tests
    success = run_command(cmd, "Running tests")
    
    if args.coverage and success:
        print(f"\n📊 Coverage report generated in htmlcov/index.html")
    
    if success:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 