#!/usr/bin/env python3
"""Debug JSON syntax error in config.v2.json"""

import json
import sys
from pathlib import Path

def find_json_error(filepath):
    """Find and report JSON syntax errors with context."""
    
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Try to parse the JSON
        json.loads(content)
        print("✓ JSON is valid!")
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON Error: {e}")
        print(f"  Location: Line {e.lineno}, Column {e.colno}")
        print(f"  Character position: {e.pos}")
        
        # Show context around the error
        lines = content.split('\n')
        error_line = e.lineno - 1  # Convert to 0-based index
        
        print("\nContext around error:")
        # Show 3 lines before and after
        start = max(0, error_line - 3)
        end = min(len(lines), error_line + 4)
        
        for i in range(start, end):
            line_num = i + 1
            prefix = ">>>" if i == error_line else "   "
            print(f"{prefix} {line_num:3d}: {lines[i][:80]}")
            
            # If this is the error line, show the column
            if i == error_line and e.colno:
                # Create pointer to the error column
                pointer = " " * (len(prefix) + 5 + e.colno) + "^"
                print(pointer)
        
        # Common fixes
        print("\nCommon JSON issues to check:")
        print("1. Missing comma between elements")
        print("2. Extra comma after last element")
        print("3. Unescaped quotes in strings")
        print("4. Unclosed brackets or braces")
        
        # Try to find specific issues
        if "Expecting ',' delimiter" in str(e):
            print("\n✓ Missing comma detected - add comma after previous element")
        elif "Expecting property name" in str(e):
            print("\n✓ Extra comma detected - remove comma before closing brace/bracket")

if __name__ == "__main__":
    config_path = Path("whisperjav/config/config.v2.json")
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    
    find_json_error(config_path)