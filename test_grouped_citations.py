#!/usr/bin/env python3

"""
Test the improved citation positioning that handles multiple citations in groups
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from filter_examples import ResponseFormatter

def test_grouped_citations():
    """Test citation formatting with multiple citations appearing together"""
    
    formatter = ResponseFormatter()
    
    # Test case: Multiple citations in one parenthetical group (like in the user's screenshot)
    print("=== Multiple Citations in One Group Test ===")
    response_text = """- Weitere Projekte/Workflows: ZotLit / Zotero‑zu‑Obsidian‑Skripte, Export via BetterBibTeX‑JSON + Obsidian‑Plugins (Zotero Integration, Templater, Dataview) zur automatischen Erzeugung/ Aktualisierung von Literature Notes. ([github.com](https://github.com/PKM-er/obsidian-zotlit?utm_source=openai), [groundwater.usu.edu](https://groundwater.usu.edu/blog/2023/A-Zotero-to-Obsidian-Workflow/?utm_source=openai))

Soll ich eine Schritt‑für‑Schritt‑Anleitung für eine dieser Optionen zusammenstellen?"""

    citations = [
        {
            'title': 'GitHub - PKM-er/obsidian-zotlit: A third-party project',
            'url': 'https://github.com/PKM-er/obsidian-zotlit?utm_source=openai',
            'snippet': 'Obsidian ZotLit Plugin'
        },
        {
            'title': 'A Zotero to Obsidian Workflow | Shuai Computational and Integrated Hydrology Group',
            'url': 'https://groundwater.usu.edu/blog/2023/A-Zotero-to-Obsidian-Workflow/?utm_source=openai',
            'snippet': 'Zotero Obsidian Workflow Guide'
        }
    ]
    
    result = formatter.format_citations(response_text, citations, "numbered", start_number=1)
    print("Input text:", response_text)
    print("\n" + "="*60)
    print("FORMATTED RESULT:")
    print("="*60)
    print(result)
    
    # Test coordinated numbering (citations starting from [3])
    print("\n\n=== Coordinated Numbering (Start from 3) ===")
    result_coordinated = formatter.format_citations(response_text, citations, "numbered", start_number=3)
    print("FORMATTED RESULT (starting from [3]):")
    print("="*60)
    print(result_coordinated)

if __name__ == "__main__":
    test_grouped_citations()
