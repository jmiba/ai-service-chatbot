#!/usr/bin/env python3

# Script to add indentation to the indexing tool section

# Read the file
with open('pages/scrape.py', 'r') as f:
    lines = f.readlines()

# Find the start and end of the indexing tool section
start_line = None
end_line = None

for i, line in enumerate(lines):
    if '# Initialize URL configs in session state' in line:
        start_line = i
    elif '# --- Show current knowledge base entries ---' in line:
        end_line = i
        break

if start_line is None or end_line is None:
    print("Could not find the indexing tool section boundaries")
    exit(1)

print(f"Found indexing tool section from line {start_line} to {end_line}")

# Add 4 spaces indentation to lines between start_line and end_line
for i in range(start_line, end_line):
    if lines[i].strip():  # Only indent non-empty lines
        lines[i] = '    ' + lines[i]

# Write the file back
with open('pages/scrape.py', 'w') as f:
    f.writelines(lines)

print("Indentation added successfully")
