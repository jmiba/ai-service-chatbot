#!/usr/bin/env python3

"""
Debug test to see exactly what's happening with citation replacement
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from filter_examples import ResponseFormatter

def debug_citation_replacement():
    """Debug the exact pattern matching and replacement"""
    
    # Exact text from the user's output
    test_text = """- Weitere Projekte/Workflows: ZotLit / Zotero‑zu‑Obsidian‑Skripte, Export via BetterBibTeX‑JSON + Obsidian‑Plugins (Zotero Integration, Templater, Dataview) zur automatischen Erzeugung/ Aktualisierung von Literature Notes. ([github.com](https://github.com/PKM-er/obsidian-zotlit?utm_source=openai), [groundwater.usu.edu](https://groundwater.usu.edu/blog/2023/A-Zotero-to-Obsidian-Workflow/?utm_source=openai))"""
    
    # Test the regex pattern
    multi_citation_pattern = r'\(\s*\[([^\]]+)\]\([^)]+\)\s*,\s*\[([^\]]+)\]\([^)]+\)\s*\)'
    
    print("=== Debug Citation Pattern Matching ===")
    print("Original text:")
    print(test_text)
    print("\nPattern:", multi_citation_pattern)
    
    matches = list(re.finditer(multi_citation_pattern, test_text))
    print(f"\nFound {len(matches)} matches:")
    for i, match in enumerate(matches):
        print(f"Match {i+1}: '{match.group(0)}'")
        print(f"  Start: {match.start()}, End: {match.end()}")
        print(f"  Group text: '{match.group(0)}'")
        print(f"  Citation count: {match.group(0).count('](')}") 
    
    # Test replacement
    def replace_multi_citation(match):
        group_text = match.group(0)
        citation_count = group_text.count('](')
        print(f"\nReplacing: '{group_text}' (count: {citation_count})")
        
        if citation_count <= 3:
            numbers = [str(1 + i) for i in range(citation_count)]
            ref_marker = f'[{",".join(numbers)}]'
        else:
            markers = [f'[{1 + i}]' for i in range(citation_count)]
            ref_marker = ''.join(markers)
        
        print(f"With: '{ref_marker}'")
        return ref_marker
    
    result = re.sub(multi_citation_pattern, replace_multi_citation, test_text)
    print(f"\nFinal result:")
    print(result)

if __name__ == "__main__":
    debug_citation_replacement()
