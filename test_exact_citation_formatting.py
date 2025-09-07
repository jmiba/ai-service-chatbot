#!/usr/bin/env python3
"""
Test Exact Real Response Formatting

Tests the citation formatting with your exact response data to verify reference markers and list formatting
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from filter_examples import ResponseFormatter
    print("✅ Successfully imported ResponseFormatter")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_exact_response():
    """Test with your exact response data"""
    print("\n🧪 Testing Exact Response Citation Formatting")
    print("=" * 50)
    
    # Your exact response text
    real_response = """Kurzüberblick zu Online‑Erwähnungen von Prof. Dr. Eduard Mühle im Zeitraum 1980–2010:

- Studium, MA 1986 und Promotion 1990; frühe Tätigkeiten (u. a. freier Mitarbeiter FAZ, Gerda‑Henkel‑Stipendium Ende 1980er). ([uni-muenster.de](https://www.uni-muenster.de/Geschichte/histsem/OE-G/muehle/index.html?utm_source=openai))  
- Tätigkeiten bei DFG (Leiter des Vorstandsreferats) und im Sekretariat der Hochschulrektorenkonferenz (1990–1995). ([uni-muenster.de](https://www.uni-muenster.de/Geschichte/histsem/OE-G/muehle/index.html?utm_source=openai), [europa-uni.de](https://www.europa-uni.de/de/universitaet/kommunikation/medienservice/medieninformationen/150-2022/index.html?utm_source=openai))  
- Direktor des Leibniz‑Instituts für historische Ostmitteleuropaforschung (Herder‑Institut) in Marburg 1995–2005 — vielfache Erwähnung in Profildarstellungen. ([uni-muenster.de](https://www.uni-muenster.de/Geschichte/histsem/OE-G/muehle/index.html?utm_source=openai), [de.wikipedia.org](https://de.wikipedia.org/wiki/Eduard_M%C3%BChle?utm_source=openai))  
- Habilitation 2004 und Schlüsselpublikationen Mitte der 2000er (z. B. 2005: Für Volk und deutschen Osten). ([de.wikipedia.org](https://de.wikipedia.org/wiki/Eduard_M%C3%BChle?utm_source=openai))  
- Ab 2008 Direktor des Deutschen Historischen Instituts Warschau (Beginn 2008 liegt noch in Ihrem Zeitraum). ([uni-muenster.de](https://www.uni-muenster.de/Geschichte/histsem/OE-G/muehle/index.html?utm_source=openai))

Hinweis: Bei der durchgeführten Web‑Suche und in den bereitgestellten Profildokumenten fanden sich überwiegend biographische Einträge, Amts‑/Berufsmitteilungen und Publikationshinweise für 1980–2010; konkrete Presse‑Newsartikel aus genau diesem Zeitraum wurden in den geprüften Quellen nicht gesondert identifiziert. ([europa-uni.de](https://www.europa-uni.de/de/universitaet/kommunikation/medienservice/medieninformationen/150-2022/index.html?utm_source=openai), [uni-muenster.de](https://www.uni-muenster.de/Geschichte/histsem/OE-G/muehle/index.html?utm_source=openai))

Soll ich gezielt in Zeitungsarchiven (z. B. FAZ‑Archiv, lokale Marburger Presse, Bibliotheksdatenbanken) nach einzelnen Presseberichten aus 1980–2010 suchen?"""

    # Your exact citations
    real_citations = [
        {
            'title': 'Abteilung für Osteuropäische Geschichte - Mühle',
            'url': 'https://www.uni-muenster.de/Geschichte/histsem/OE-G/muehle/index.html?utm_source=openai'
        },
        {
            'title': 'Medieninformation Nr. 150-2022 • Europa-Universität Viadrina',
            'url': 'https://www.europa-uni.de/de/universitaet/kommunikation/medienservice/medieninformationen/150-2022/index.html?utm_source=openai'
        },
        {
            'title': 'Eduard Mühle',
            'url': 'https://de.wikipedia.org/wiki/Eduard_M%C3%BChle?utm_source=openai'
        }
    ]
    
    print("🔍 Original response:")
    print(f"   Length: {len(real_response)} chars")
    print(f"   Number of citation groups: {real_response.count('([')}")
    
    # Apply Numbered formatting
    print("\n📝 Applying Numbered Citation Formatting...")
    formatter = ResponseFormatter("Numbered")
    formatted = formatter.format_citations(real_response, real_citations)
    
    print(f"\n✅ Formatted response:")
    print("=" * 60)
    print(formatted)
    print("=" * 60)
    
    # Analyze results
    has_inline_removed = "([uni-muenster.de]" not in formatted
    has_reference_markers = "[1" in formatted.split("**References:**")[0] if "**References:**" in formatted else False
    references_section = formatted.split("**References:**")[1] if "**References:**" in formatted else ""
    has_proper_list = references_section.count('\n[1]') == 1 and references_section.count('\n[2]') == 1 and references_section.count('\n[3]') == 1
    
    print(f"\n🔍 Analysis:")
    print(f"   ✅ Inline citations removed: {has_inline_removed}")
    print(f"   ✅ Reference markers in text: {has_reference_markers}")
    print(f"   ✅ Proper references list: {has_proper_list}")
    print(f"   ✅ References section exists: {'**References:**' in formatted}")
    
    if has_reference_markers:
        # Find where the reference marker was inserted
        main_text = formatted.split("**References:**")[0]
        marker_pos = -1
        for marker in ['[1-3]', '[1,2,3]', '[1]']:
            if marker in main_text:
                marker_pos = main_text.find(marker)
                print(f"   📍 Reference marker '{marker}' found at position {marker_pos}")
                break
    
    return has_inline_removed and has_reference_markers and has_proper_list

if __name__ == "__main__":
    print("🧪 Exact Real Response Citation Formatting Test")
    print("=" * 60)
    
    success = test_exact_response()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Perfect! Citation formatting now includes:")
        print("   ✅ Reference markers in the main text")
        print("   ✅ Clean removal of inline citations")
        print("   ✅ Proper numbered references list")
        print("   💡 Ready for your Streamlit app!")
    else:
        print("❌ Still needs some fine-tuning...")
