#!/usr/bin/env python3
"""
Test Improved Web Citation Formatting

Tests the improved citation formatting with real response data
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

def test_real_response_formatting():
    """Test with the actual response from your debug dump"""
    print("\n🧪 Testing Real Response Formatting")
    print("=" * 50)
    
    # Actual response from the debug dump
    real_response = "Sie stammt aus Brandenburg an der Havel; dort wurde sie 1982 geboren. ([sahra-damus.de](https://www.sahra-damus.de/wordpress/ueber-mich/?utm_source=openai), [de.wikipedia.org](https://de.wikipedia.org/wiki/Sahra_Damus?utm_source=openai))"
    
    # Actual citations from URL annotations
    real_citations = [
        {
            'title': 'Sahra Damus',
            'url': 'https://www.sahra-damus.de/wordpress/ueber-mich/?utm_source=openai'
        },
        {
            'title': 'Sahra Damus',
            'url': 'https://de.wikipedia.org/wiki/Sahra_Damus?utm_source=openai'
        }
    ]
    
    print("🔍 Original response:")
    print(f"   {real_response}")
    print(f"   Length: {len(real_response)} chars")
    
    # Test Numbered formatting
    formatter = ResponseFormatter("Numbered")
    formatted = formatter.format_citations(real_response, real_citations)
    
    print("\n🔍 Numbered formatting result:")
    print(f"   {formatted}")
    print(f"   Length: {len(formatted)} chars")
    
    # Check if inline citations were removed
    has_inline_removed = "[sahra-damus.de]" not in formatted
    has_references_added = "**References:**" in formatted
    
    print(f"\n✅ Inline citations removed: {has_inline_removed}")
    print(f"✅ References section added: {has_references_added}")
    
    # Test other formats
    print("\n🔍 Testing other formats:")
    for style in ["Academic (APA-style)", "Simple links"]:
        formatter = ResponseFormatter(style)
        result = formatter.format_citations(real_response, real_citations)
        clean_text = result.replace(real_response.split("(")[0], "").strip()
        print(f"   {style}: Added {len(clean_text)} chars of formatting")
    
    return has_inline_removed and has_references_added

if __name__ == "__main__":
    print("🧪 Improved Web Citation Formatting Test")
    print("=" * 50)
    
    success = test_real_response_formatting()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Citation formatting improvements working!")
        print("💡 The inline citations should now be properly removed and replaced with clean references.")
    else:
        print("❌ Citation formatting needs more work.")
