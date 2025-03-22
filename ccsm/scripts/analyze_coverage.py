#!/usr/bin/env python3
"""
CCSM C++ Test Coverage Analysis Script

This script analyzes LCOV coverage reports and provides a summary of the
test coverage for each component of the CCSM C++ codebase. It also
identifies areas with low coverage that need improvement.

Usage:
    python analyze_coverage.py [--html] [coverage_info_file]

Arguments:
    --html:            Generate an HTML report with detailed coverage information
    coverage_info_file: Path to the lcov.info file (default: build/coverage.info)
"""

import sys
import os
import re
import json
import argparse
import subprocess
from collections import defaultdict

# Components and their corresponding files
COMPONENTS = {
    "Core Tensor System": [r"src/tensor\.(cpp|h)$"],
    "GGML Backend": [r"src/cpu/ggml_tensor\.(cpp|h)$", r"src/cpu/ggml_model\.(cpp|h)$"],
    "MLX Backend": [r"src/mlx/.*\.(cpp|h)$"],
    "Model System": [r"src/model\.(cpp|h)$", r"src/model_loader\.(cpp|h)$"],
    "Tokenizer": [r"src/tokenizer\.(cpp|h)$"],
    "Generator": [r"src/generator\.(cpp|h)$"],
    "Watermarking": [r"src/watermarking\.(cpp|h)$"],
    "Thread Pool": [r"src/cpu/thread_pool\.(cpp|h)$"],
    "SIMD Optimizations": [r"src/cpu/simd\.(cpp|h)$"],
    "Command-line Arguments": [r"src/cli_args\.(cpp|h)$"],
    "Utility Functions": [r"src/utils\.(cpp|h)$"],
}

def parse_lcov_info(lcov_file):
    """Parse an LCOV coverage info file."""
    coverage_data = {}
    current_file = None
    
    with open(lcov_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('SF:'):
                # Start of a new file
                current_file = line[3:]
                coverage_data[current_file] = {
                    'lines_total': 0,
                    'lines_covered': 0,
                    'branches_total': 0,
                    'branches_covered': 0,
                    'functions_total': 0,
                    'functions_covered': 0,
                    'line_coverage': {},
                }
            elif line.startswith('DA:'):
                # Line coverage data: DA:<line number>,<execution count>
                parts = line[3:].split(',')
                line_num = int(parts[0])
                count = int(parts[1])
                coverage_data[current_file]['line_coverage'][line_num] = count
                coverage_data[current_file]['lines_total'] += 1
                if count > 0:
                    coverage_data[current_file]['lines_covered'] += 1
            elif line.startswith('BRDA:'):
                # Branch coverage data: BRDA:<line number>,<block number>,<branch number>,<taken>
                parts = line[5:].split(',')
                taken = parts[3]
                coverage_data[current_file]['branches_total'] += 1
                if taken != '-' and int(taken) > 0:
                    coverage_data[current_file]['branches_covered'] += 1
            elif line.startswith('FN:'):
                # Function definition: FN:<line number>,<function name>
                coverage_data[current_file]['functions_total'] += 1
            elif line.startswith('FNDA:'):
                # Function execution data: FNDA:<execution count>,<function name>
                parts = line[5:].split(',')
                count = int(parts[0])
                if count > 0:
                    coverage_data[current_file]['functions_covered'] += 1
    
    return coverage_data

def get_component_for_file(file_path):
    """Determine which component a file belongs to."""
    for component, patterns in COMPONENTS.items():
        for pattern in patterns:
            if re.search(pattern, file_path):
                return component
    return "Other"

def calculate_coverage_by_component(coverage_data):
    """Calculate coverage percentage for each component."""
    component_coverage = {}
    
    # Initialize component data
    for component in COMPONENTS:
        component_coverage[component] = {
            'lines_total': 0,
            'lines_covered': 0,
            'branches_total': 0,
            'branches_covered': 0,
            'functions_total': 0,
            'functions_covered': 0,
            'files': [],
        }
    
    component_coverage["Other"] = {
        'lines_total': 0,
        'lines_covered': 0,
        'branches_total': 0,
        'branches_covered': 0,
        'functions_total': 0,
        'functions_covered': 0,
        'files': [],
    }
    
    # Aggregate data by component
    for file_path, file_data in coverage_data.items():
        component = get_component_for_file(file_path)
        component_coverage[component]['lines_total'] += file_data['lines_total']
        component_coverage[component]['lines_covered'] += file_data['lines_covered']
        component_coverage[component]['branches_total'] += file_data['branches_total']
        component_coverage[component]['branches_covered'] += file_data['branches_covered']
        component_coverage[component]['functions_total'] += file_data['functions_total']
        component_coverage[component]['functions_covered'] += file_data['functions_covered']
        component_coverage[component]['files'].append({
            'path': file_path,
            'lines_total': file_data['lines_total'],
            'lines_covered': file_data['lines_covered'],
            'line_coverage': file_data['line_coverage'],
            'line_percent': (file_data['lines_covered'] / file_data['lines_total'] * 100) if file_data['lines_total'] > 0 else 0,
        })
    
    # Calculate percentages
    for component, data in component_coverage.items():
        if data['lines_total'] > 0:
            data['line_percent'] = data['lines_covered'] / data['lines_total'] * 100
        else:
            data['line_percent'] = 0
            
        if data['branches_total'] > 0:
            data['branch_percent'] = data['branches_covered'] / data['branches_total'] * 100
        else:
            data['branch_percent'] = 0
            
        if data['functions_total'] > 0:
            data['function_percent'] = data['functions_covered'] / data['functions_total'] * 100
        else:
            data['function_percent'] = 0
    
    return component_coverage

def find_low_coverage_areas(coverage_data, threshold=50):
    """Find files and functions with coverage below the threshold."""
    low_coverage = []
    
    for file_path, file_data in coverage_data.items():
        if file_data['lines_total'] > 0:
            line_percent = file_data['lines_covered'] / file_data['lines_total'] * 100
            if line_percent < threshold:
                low_coverage.append({
                    'file': file_path,
                    'coverage': line_percent,
                    'covered_lines': file_data['lines_covered'],
                    'total_lines': file_data['lines_total'],
                })
    
    return sorted(low_coverage, key=lambda x: x['coverage'])

def identify_uncovered_functions(coverage_data):
    """Identify functions with zero coverage."""
    # This requires parsing the source files and matching with coverage data
    # For simplicity, we'll just show files with low coverage
    return []

def print_coverage_summary(component_coverage):
    """Print a coverage summary to the console."""
    # Calculate total coverage
    total_lines = sum(data['lines_total'] for data in component_coverage.values())
    total_covered = sum(data['lines_covered'] for data in component_coverage.values())
    overall_percent = (total_covered / total_lines * 100) if total_lines > 0 else 0
    
    print(f"\nCCSM C++ Test Coverage Summary\n")
    print(f"Overall coverage: {overall_percent:.1f}% ({total_covered}/{total_lines} lines)\n")
    
    print(f"{'Component':<25} {'Coverage':<10} {'Lines':<15} {'Status'}")
    print(f"{'-'*25} {'-'*10} {'-'*15} {'-'*10}")
    
    # Print component coverage sorted by coverage percentage
    for component, data in sorted(component_coverage.items(), 
                                 key=lambda x: x[1]['line_percent'] if x[1]['lines_total'] > 0 else 0, 
                                 reverse=True):
        if data['lines_total'] == 0:
            status = "N/A"
        elif data['line_percent'] >= 80:
            status = "✅ Good"
        elif data['line_percent'] >= 50:
            status = "🟡 Partial"
        else:
            status = "🔴 Low"
            
        print(f"{component:<25} {data['line_percent']:>6.1f}%  {data['lines_covered']}/{data['lines_total']:<8} {status}")
    
    print("\n")

def print_low_coverage_details(low_coverage_areas):
    """Print details about areas with low coverage."""
    if not low_coverage_areas:
        print("No areas with particularly low coverage detected.")
        return
    
    print(f"Files with low coverage (<50%):\n")
    print(f"{'File':<50} {'Coverage':<10} {'Lines'}")
    print(f"{'-'*50} {'-'*10} {'-'*15}")
    
    for area in low_coverage_areas:
        print(f"{area['file']:<50} {area['coverage']:>6.1f}%  {area['covered_lines']}/{area['total_lines']}")
    
    print("\n")

def generate_html_report(component_coverage, low_coverage_areas, output_dir="coverage_report"):
    """Generate an HTML report with detailed coverage information."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total coverage
    total_lines = sum(data['lines_total'] for data in component_coverage.values())
    total_covered = sum(data['lines_covered'] for data in component_coverage.values())
    overall_percent = (total_covered / total_lines * 100) if total_lines > 0 else 0
    
    # Write main index.html
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>CCSM C++ Test Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .good {{ color: green; }}
        .partial {{ color: orange; }}
        .low {{ color: red; }}
        .progress-container {{ width: 100px; background-color: #f1f1f1; }}
        .progress-bar {{ height: 20px; }}
        .good-bar {{ background-color: #4CAF50; }}
        .partial-bar {{ background-color: #FFA500; }}
        .low-bar {{ background-color: #f44336; }}
    </style>
</head>
<body>
    <h1>CCSM C++ Test Coverage Report</h1>
    <p><strong>Overall coverage: {overall_percent:.1f}%</strong> ({total_covered}/{total_lines} lines)</p>
    
    <h2>Component Coverage</h2>
    <table>
        <tr>
            <th>Component</th>
            <th>Coverage</th>
            <th>Lines</th>
            <th>Status</th>
            <th>Progress</th>
        </tr>
""")
        
        # Add component rows, sorted by coverage
        for component, data in sorted(component_coverage.items(), 
                                     key=lambda x: x[1]['line_percent'] if x[1]['lines_total'] > 0 else 0, 
                                     reverse=True):
            if data['lines_total'] == 0:
                status = "N/A"
                status_class = ""
                bar_class = ""
            elif data['line_percent'] >= 80:
                status = "Good"
                status_class = "good"
                bar_class = "good-bar"
            elif data['line_percent'] >= 50:
                status = "Partial"
                status_class = "partial"
                bar_class = "partial-bar"
            else:
                status = "Low"
                status_class = "low"
                bar_class = "low-bar"
                
            f.write(f"""
        <tr>
            <td><a href="component_{component.replace(' ', '_')}.html">{component}</a></td>
            <td>{data['line_percent']:.1f}%</td>
            <td>{data['lines_covered']}/{data['lines_total']}</td>
            <td class="{status_class}">{status}</td>
            <td>
                <div class="progress-container">
                    <div class="progress-bar {bar_class}" style="width: {min(100, data['line_percent'])}%;"></div>
                </div>
            </td>
        </tr>""")
        
        f.write("""
    </table>
    
    <h2>Files with Low Coverage</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Coverage</th>
            <th>Lines</th>
        </tr>
""")
        
        # Add low coverage files
        for area in low_coverage_areas:
            f.write(f"""
        <tr>
            <td>{area['file']}</td>
            <td>{area['coverage']:.1f}%</td>
            <td>{area['covered_lines']}/{area['total_lines']}</td>
        </tr>""")
            
        f.write("""
    </table>
</body>
</html>
""")
    
    # Create component detail pages
    for component, data in component_coverage.items():
        component_filename = f"component_{component.replace(' ', '_')}.html"
        with open(os.path.join(output_dir, component_filename), 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{component} - CCSM C++ Test Coverage</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .progress-container {{ width: 100px; background-color: #f1f1f1; }}
        .progress-bar {{ height: 20px; }}
        .good-bar {{ background-color: #4CAF50; }}
        .partial-bar {{ background-color: #FFA500; }}
        .low-bar {{ background-color: #f44336; }}
    </style>
</head>
<body>
    <h1>{component} Coverage Details</h1>
    <p><a href="index.html">← Back to Summary</a></p>
    
    <p><strong>Overall component coverage: {data['line_percent']:.1f}%</strong> ({data['lines_covered']}/{data['lines_total']} lines)</p>
    <p>Functions: {data['functions_covered']}/{data['functions_total']} ({data['function_percent']:.1f}%)</p>
    <p>Branches: {data['branches_covered']}/{data['branches_total']} ({data['branch_percent']:.1f}%)</p>
    
    <h2>Files</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Coverage</th>
            <th>Lines</th>
            <th>Progress</th>
        </tr>
""")
            
            # Add file rows, sorted by coverage
            for file_data in sorted(data['files'], key=lambda x: x['line_percent'], reverse=True):
                if file_data['line_percent'] >= 80:
                    bar_class = "good-bar"
                elif file_data['line_percent'] >= 50:
                    bar_class = "partial-bar"
                else:
                    bar_class = "low-bar"
                    
                f.write(f"""
        <tr>
            <td>{file_data['path']}</td>
            <td>{file_data['line_percent']:.1f}%</td>
            <td>{file_data['lines_covered']}/{file_data['lines_total']}</td>
            <td>
                <div class="progress-container">
                    <div class="progress-bar {bar_class}" style="width: {min(100, file_data['line_percent'])}%;"></div>
                </div>
            </td>
        </tr>""")
                
            f.write("""
    </table>
</body>
</html>
""")
    
    print(f"HTML report generated in {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Analyze CCSM C++ test coverage')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('coverage_file', nargs='?', default='build/coverage.info', 
                      help='Path to lcov.info file (default: build/coverage.info)')
    
    args = parser.parse_args()
    
    # Check if coverage file exists
    if not os.path.exists(args.coverage_file):
        print(f"Error: Coverage file '{args.coverage_file}' not found.")
        print("Please run the coverage build and tests first:")
        print("  ./build.sh --coverage")
        print("  cd build && ctest -V")
        print("  cd build && make coverage")
        return 1
    
    # Parse coverage data
    coverage_data = parse_lcov_info(args.coverage_file)
    
    # Calculate coverage by component
    component_coverage = calculate_coverage_by_component(coverage_data)
    
    # Find areas with low coverage
    low_coverage_areas = find_low_coverage_areas(coverage_data)
    
    # Print summary to console
    print_coverage_summary(component_coverage)
    print_low_coverage_details(low_coverage_areas)
    
    # Generate HTML report if requested
    if args.html:
        generate_html_report(component_coverage, low_coverage_areas)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())