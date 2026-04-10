#!/usr/bin/env python3
"""
STEP File Number Rounding Script
================================

This script rounds floating-point numbers in STEP files to 6 decimal places.
It processes both single STEP files and directories containing STEP files.

Usage:
    python round_step_numbers.py <input_path>

Input path can be:
1. A single .step file
2. A directory containing subdirectories with .step files

# Process a single file
python round_step_numbers.py data/reordered_step/0002/00020000/00020000_7259f13b84f34cce843db1f8_step_043.step

# Process an entire directory
python round_step_numbers.py data/reordered_step/0002

# Process the entire reordered_step directory
python round_step_numbers.py data/reordered_step

Output:
    Processed files are saved under ./data/reorder_round_step (configurable via --output-dir)
    maintaining the same directory structure as input.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Union, List

# STEP string literals: single-quoted, '' is the escaped quote.
_STR_LIT_RE = re.compile(r"'(?:[^']|'')*'")


def _sub_outside_strings(pattern, repl, text: str) -> str:
    """Apply re.sub only to spans outside STEP string literals."""
    out, last = [], 0
    for sm in _STR_LIT_RE.finditer(text):
        out.append(re.sub(pattern, repl, text[last:sm.start()]))
        out.append(sm.group(0))
        last = sm.end()
    out.append(re.sub(pattern, repl, text[last:]))
    return "".join(out)


def round_float_numbers(content: str) -> str:
    """
    Round floating-point numbers in STEP file content to 6 decimal places.
    
    Args:
        content: The STEP file content as a string
        
    Returns:
        Modified content with rounded numbers
    """
    # Pattern to match floating-point numbers in various formats:
    # - Regular decimals: 0.123456789
    # - Scientific notation: 1.23E-17
    # - Negative numbers: -0.123456789
    # - Numbers in parentheses: ( 0.123456789, -0.456789 )
    # - Numbers with trailing zeros: 0.123456789000000
    
    # ISO 10303-21 §6.4.3 REAL literals: optional sign, digits, '.', optional
    # fractional digits, optional exponent (E or e). Forms: 1.5, 1., .5, 1.5E-3.
    # No \b — '.' breaks word-boundary semantics on the trailing-dot form.
    float_pattern = r'-?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?'
    
    def round_match(match):
        number_str = match.group(0)

        # Skip entity refs like #123. match.string is the segment this regex
        # is currently running over (per re.sub semantics) — match.start() is
        # relative to that segment, not the outer `content`.
        if match.start() > 0 and match.string[match.start()-1] == '#':
            return number_str

        try:
            if 'E' in number_str or 'e' in number_str:
                norm = number_str.replace('e', 'E')
                mantissa, exponent = norm.split('E', 1)
                rounded_mantissa = round(float(mantissa), 6)
                return f"{rounded_mantissa:.6f}E{exponent}"
            else:
                rounded = round(float(number_str), 6)
                return f"{rounded:.6f}"
        except ValueError:
            return number_str
    
    # Apply rounding only outside string literals — PRODUCT('Version 2.1234567')
    # is metadata, not geometry; rounding it would corrupt the part name.
    return _sub_outside_strings(float_pattern, round_match, content)

def process_single_step_file(input_path: str, output_dir: str) -> None:
    """
    Process a single STEP file.
    
    Args:
        input_path: Path to the input STEP file
        output_dir: Output directory path
    """
    input_path = Path(input_path)
    output_path = Path(output_dir) / input_path.name
    
    print(f"Processing: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Round the numbers
        rounded_content = round_float_numbers(content)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the processed content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rounded_content)
        
        print(f"Saved: {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Process all STEP files in a directory and its subdirectories.
    
    Args:
        input_dir: Path to the input directory
        output_dir: Output directory path
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"Processing directory: {input_path}")
    
    # Find all .step files recursively
    step_files = list(input_path.rglob("*.step"))
    
    if not step_files:
        print(f"No .step files found in {input_path}")
        return
    
    print(f"Found {len(step_files)} STEP files")
    
    for step_file in step_files:
        # Calculate relative path from input directory
        rel_path = step_file.relative_to(input_path)
        
        # Create corresponding output path
        output_file_path = output_path / rel_path
        
        print(f"Processing: {step_file}")
        
        try:
            with open(step_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Round the numbers
            rounded_content = round_float_numbers(content)
            
            # Create output directory if it doesn't exist
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the processed content
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(rounded_content)
            
            print(f"Saved: {output_file_path}")
            
        except Exception as e:
            print(f"Error processing {step_file}: {e}")

def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Round floating-point numbers in STEP files to 6 decimal places"
    )
    parser.add_argument(
        "input_path",
        help="Path to a single STEP file or directory containing STEP files"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/reorder_round_step",
        help="Output directory (default: ./data/reorder_round_step)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    # Validate input path
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process based on input type
    if input_path.is_file():
        if input_path.suffix.lower() != '.step':
            print(f"Error: Input file is not a .step file: {input_path}")
            sys.exit(1)
        process_single_step_file(str(input_path), str(output_dir))
    elif input_path.is_dir():
        process_directory(str(input_path), str(output_dir))
    else:
        print(f"Error: Input path is neither a file nor directory: {input_path}")
        sys.exit(1)
    
    print("Processing completed!")

if __name__ == "__main__":
    main() 