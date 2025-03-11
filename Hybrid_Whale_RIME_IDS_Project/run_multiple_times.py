#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys

def run_script(script_path, count):
    """
    Executes the specified Python script 'count' times using the current Python interpreter.
    """
    for i in range(1, count + 1):
        print(f"Iteration {i} of {count}: Executing {script_path}")
        try:
            # Use the current Python interpreter to run the target script
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error on iteration {i}: {e}")
        except Exception as e:
            print(f"Unexpected error on iteration {i}: {e}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a specified Python script a given number of times.")
    parser.add_argument("script_path", help="Path to the Python script to be executed")
    parser.add_argument("count", type=int, help="Number of times to run the script")
    args = parser.parse_args()

    # Check if the provided script exists
    if not os.path.isfile(args.script_path):
        print(f"Error: The file '{args.script_path}' does not exist.")
        sys.exit(1)

    run_script(args.script_path, args.count)

if __name__ == "__main__":
    main()
