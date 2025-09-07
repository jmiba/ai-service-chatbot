#!/usr/bin/env python3

"""
Test script to debug citation formatting issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from filter_examples import ResponseFormatter

def test_numbered_citations():
    """Test numbered citation formatting with different scenarios"""
    
    formatter = ResponseFormatter()
    
    # Test case 1: Single citation
    print("=== Test 1: Single Citation ===")
    response_text = "This is about Zotero, a free literature management program."
    citations = [
        {
            'title': 'Zotero - Your personal research assistant',
            'url': 'https://www.zotero.org',
            'snippet': 'Zotero is a free, easy-to-use tool to help you collect, organize, cite, and share research.'
        }
    ]
    
    result = formatter.format_citations(response_text, citations, "numbered", start_number=1)
    print("Input:", response_text)
    print("Result:", result)
    print()
    
    # Test case 2: Multiple citations (should show individual markers, not range)
    print("=== Test 2: Multiple Citations ===")
    response_text = "There are several Zotero plugins for syncing with Notion, including community solutions and third-party integrations."
    citations = [
        {
            'title': 'Notero: A Zotero plugin for syncing items and notes into Notion',
            'url': 'https://github.com/dvanoni/notero',
            'snippet': 'GitHub repository for Notero plugin'
        },
        {
            'title': 'A Zotero plugin for syncing items and notes into Notion',
            'url': 'https://github.com/makinteract/notero-notes',
            'snippet': 'Alternative Notero implementation'
        },
        {
            'title': 'ZoteroToNotion: Sync Zotero files to Notion',
            'url': 'https://github.com/nanbhas/ZoteroToNotion',
            'snippet': 'Another Zotero to Notion sync solution'
        }
    ]
    
    result = formatter.format_citations(response_text, citations, "numbered", start_number=1)
    print("Input:", response_text)
    print("Result:", result)
    print()
    
    # Test case 3: Coordinated numbering (web citations starting after file citations)
    print("=== Test 3: Coordinated Numbering (Start from 4) ===")
    result = formatter.format_citations(response_text, citations, "numbered", start_number=4)
    print("Input:", response_text)
    print("Result:", result)
    print()
    
    # Test case 4: Many citations (6 citations like in screenshot)
    print("=== Test 4: Six Citations ===")
    response_text = "Zotero offers multiple integration options with various note-taking and reference management systems."
    citations = [
        {'title': 'Notero Plugin', 'url': 'https://github.com/dvanoni/notero'},
        {'title': 'Notero Notes', 'url': 'https://github.com/makinteract/notero-notes'},
        {'title': 'ZoteroToNotion', 'url': 'https://github.com/nanbhas/ZoteroToNotion'},
        {'title': 'Make Integromat Solution', 'url': 'https://www.simonesmerilli.com/life/zotero-notion'},
        {'title': 'Alternative Notero', 'url': 'https://github.com/dvanoni/notero'},
        {'title': 'Notion Import Help', 'url': 'https://www.notion.com/help/import-data-into-notion'}
    ]
    
    result = formatter.format_citations(response_text, citations, "numbered", start_number=1)
    print("Input:", response_text)
    print("Result:", result)
    print()

if __name__ == "__main__":
    test_numbered_citations()
