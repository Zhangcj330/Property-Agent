#!/usr/bin/env python
"""
Run API endpoint tests for Property Agent backend.
"""
import os
import sys
import pytest

def main():
    """Run the tests with proper arguments"""
    # Add the parent directory to sys.path so that the app module can be imported
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Define test files to run
    test_files = [
        'tests/test_api_endpoints.py',
        'tests/test_property_services.py',
        'tests/test_integration.py'
    ]
    
    # Run the tests with verbose output
    args = [
        '-v',                  # Verbose output
        '--asyncio-mode=auto',  # Handle async tests automatically
    ] + test_files
    
    # Run pytest with configured arguments
    print("Running tests without coverage...")
    return pytest.main(args)

if __name__ == '__main__':
    sys.exit(main()) 