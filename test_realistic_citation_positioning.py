#!/usr/bin/env python3

"""
Test script to specifically test the positioning of citation markers in realistic text
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from filter_examples import ResponseFormatter

def test_realistic_citation_positioning():
    """Test citation formatting with realistic text that matches the user's screenshot"""
    
    formatter = ResponseFormatter()
    
    # Test case: Text similar to the user's actual response with inline citations
    print("=== Realistic Citation Positioning Test ===")
    response_text = """Kurz: Ja — Zotero lässt sich mit Obsidian verbinden; es gibt mehrere gebräuchliche Wege und Plugins.

- Offizielle Grundlage: Zotero als Literaturverwaltungsprogramm (Desktop + Web + Connector).   
- Obsidian‑Community‑Plugin „Zotero Integration" (Import von Zitaten, Bibliographien, Notizen, PDF‑Annotationen; nutzt oft Better BibTeX in Zotero). ([github.com](https://github.com/mgmeyers/obsidian-zotero-integration?utm_source=openai))  
- Mdnotes (Zotero‑Addon): exportiert Einträge, Notizen und Annotationen als Markdown direkt in dein Obsidian‑Vault. Anleitung/Setups sind verbreitet. ([github.com](https://github.com/ishnid/zotero-obsidian?utm_source=openai))  
- Weitere Projekte/Workflows: ZotLit / Zotero‑zu‑Obsidian‑Skripte, Export via BetterBibTeX‑JSON + Obsidian‑Plugins (Zotero Integration, Templater, Dataview) zur automatischen Erzeugung/ Aktualisierung von Literature Notes. ([github.com](https://github.com/PKM-er/obsidian-zotlit?utm_source=openai), [groundwater.usu.edu](https://groundwater.usu.edu/blog/2023/A-Zotero-to-Obsidian-Workflow/?utm_source=openai))  
- Hinweise / Einschränkungen: Anhänge/PDFs und automatische bidirektionale Synchronisation sind teilweise eingeschränkt; manche Tools können Notizen überschreiben; Plugins sind Dritt‑Entwicklungen und können bei Updates Probleme machen (Forenmeldungen / Bugreports existieren). ([forum.obsidian.md](https://forum.obsidian.md/t/mdnotes-plugin-for-zotero-not-creating-note-anymore/23529?utm_source=openai))

Soll ich eine Schritt‑für‑Schritt‑Anleitung für eine dieser Optionen (z. B. Mdnotes oder das Obsidian‑Plugin „Zotero Integration") zusammenstellen?"""

    citations = [
        {
            'title': 'GitHub - mgmeyers/obsidian-zotero-integration: Insert and import citations, bibliographies, notes, and PDF annotations from Zotero into Obsidian.',
            'url': 'https://github.com/mgmeyers/obsidian-zotero-integration?utm_source=openai',
            'snippet': 'Obsidian Zotero Integration Plugin'
        },
        {
            'title': 'GitHub - ishnid/zotero-obsidian',
            'url': 'https://github.com/ishnid/zotero-obsidian?utm_source=openai',
            'snippet': 'Zotero Obsidian Integration'
        },
        {
            'title': 'GitHub - PKM-er/obsidian-zotlit: A third-party project that aims to facilitate the integration between Obsidian.md and Zotero',
            'url': 'https://github.com/PKM-er/obsidian-zotlit?utm_source=openai',
            'snippet': 'Obsidian ZotLit Plugin'
        },
        {
            'title': 'A Zotero to Obsidian Workflow | Shuai Computational and Integrated Hydrology Group',
            'url': 'https://groundwater.usu.edu/blog/2023/A-Zotero-to-Obsidian-Workflow/?utm_source=openai',
            'snippet': 'Zotero Obsidian Workflow Guide'
        },
        {
            'title': 'Mdnotes plugin for Zotero not creating note anymore - Help - Obsidian Forum',
            'url': 'https://forum.obsidian.md/t/mdnotes-plugin-for-zotero-not-creating-note-anymore/23529?utm_source=openai',
            'snippet': 'Obsidian Forum Discussion'
        }
    ]
    
    result = formatter.format_citations(response_text, citations, "numbered", start_number=1)
    print("Input text (truncated):", response_text[:200] + "...")
    print("\n" + "="*80)
    print("FORMATTED RESULT:")
    print("="*80)
    print(result)
    print("\n" + "="*80)
    
    # Also test coordinated numbering (starting from 4)
    print("\n=== Coordinated Numbering Test (Start from 4) ===")
    result_coordinated = formatter.format_citations(response_text, citations, "numbered", start_number=4)
    print("FORMATTED RESULT (starting from [4]):")
    print("="*80)
    print(result_coordinated[:500] + "..." if len(result_coordinated) > 500 else result_coordinated)

if __name__ == "__main__":
    test_realistic_citation_positioning()
