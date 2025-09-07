#!/usr/bin/env python3
"""
Test Web Citation Formatting

This test verifies that web citation formatting works with real-world examples
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

def test_web_citation_formatting():
    """Test web citation formatting with realistic examples"""
    print("\n🧪 Testing Web Citation Formatting")
    print("=" * 50)
    
    # Test case similar to your screenshot
    test_response = """Ja. Sahra Damus war Mitglied des Landtags Brandenburg von 2019 bis 2024. (de.wikipedia.org, sahra-damus.de)

An der Viadrina ist/war sie außerdem als Referentin für Nachhaltigkeit im Kanzlerbüro, als Anti-Korruptionsbeauftragte und als Koordinatorin „Sozialer Wandel in der Region" geführt.

Sie hat zur Landtagswahl 2024 nicht erneut kandidiert. (moz.de, sahra-damus.de)"""

    # Test different citation styles
    styles = ["Numbered", "Academic (APA-style)", "Simple links"]
    
    for style in styles:
        print(f"\n🔍 Testing {style} formatting:")
        print("-" * 30)
        
        formatter = ResponseFormatter(style)
        
        # Extract citations from text (simulating the extraction function)
        test_citations = [
            {"title": "de.wikipedia.org", "url": "https://de.wikipedia.org"},
            {"title": "sahra-damus.de", "url": "https://sahra-damus.de"},
            {"title": "moz.de", "url": "https://moz.de"}
        ]
        
        formatted = formatter.format_citations(test_response, test_citations)
        
        print("Original length:", len(test_response))
        print("Formatted length:", len(formatted))
        print("Has inline citations removed:", "(de.wikipedia.org" not in formatted)
        print("Has new formatting added:", len(formatted) > len(test_response.replace("(de.wikipedia.org, sahra-damus.de)", "").replace("(moz.de, sahra-damus.de)", "")))
        
        # Show a preview
        if len(formatted) > 200:
            print("Preview:", formatted[:200] + "...")
        else:
            print("Full result:", formatted)

def test_citation_extraction():
    """Test the citation extraction from text"""
    print("\n🧪 Testing Citation Extraction from Text")
    print("=" * 50)
    
    # Import the extraction function
    sys.path.insert(0, str(project_root))
    from app import extract_web_citations_from_text
    
    test_text = "Some text (de.wikipedia.org, sahra-damus.de) and more text (moz.de, example.org)"
    
    citations = extract_web_citations_from_text(test_text)
    
    print(f"Found {len(citations)} citations:")
    for i, citation in enumerate(citations, 1):
        print(f"  {i}. {citation['title']} -> {citation['url']}")
    
    return len(citations) > 0

if __name__ == "__main__":
    print("🧪 Web Citation Formatting Test")
    print("=" * 50)
    
    test_web_citation_formatting()
    success = test_citation_extraction()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Web citation formatting test completed!")
        print("💡 Now test in your app with debug mode enabled to see the formatting in action.")
    else:
        print("❌ Some tests failed.")
