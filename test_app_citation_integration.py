#!/usr/bin/env python3
"""
Test App Citation Integration

Verifies that the app.py citation integration is working properly
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_citation_extraction():
    """Test citation extraction from OpenAI response"""
    print("🧪 Testing Citation Extraction")
    print("=" * 50)
    
    # Import the function from app.py
    try:
        from app import extract_web_search_citations
        print("✅ Successfully imported extract_web_search_citations from app.py")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Mock response with url_citation annotations like from your debug dump
    mock_response = {
        'choices': [{
            'message': {
                'content': 'Sie stammt aus Brandenburg an der Havel; dort wurde sie 1982 geboren.',
                'tool_calls': [{
                    'function': {
                        'name': 'web_search'
                    }
                }]
            }
        }],
        'url_citation': [
            {
                'title': 'Sahra Damus',
                'url': 'https://www.sahra-damus.de/wordpress/ueber-mich/?utm_source=openai'
            },
            {
                'title': 'Sahra Damus', 
                'url': 'https://de.wikipedia.org/wiki/Sahra_Damus?utm_source=openai'
            }
        ]
    }
    
    # Test citation extraction
    citations = extract_web_search_citations(mock_response)
    
    print(f"📊 Extracted {len(citations)} citations:")
    for i, citation in enumerate(citations, 1):
        print(f"   [{i}] {citation['title']} - {citation['url']}")
    
    # Verify extraction worked
    expected_count = 2
    has_titles = all('title' in c for c in citations)
    has_urls = all('url' in c for c in citations)
    
    print(f"\n✅ Expected count ({expected_count}): {len(citations) == expected_count}")
    print(f"✅ All have titles: {has_titles}")
    print(f"✅ All have URLs: {has_urls}")
    
    return len(citations) == expected_count and has_titles and has_urls

def test_filter_loading():
    """Test filter settings loading"""
    print("\n🧪 Testing Filter Settings Loading")
    print("=" * 50)
    
    try:
        from app import get_filter_settings
        print("✅ Successfully imported get_filter_settings from app.py")
        
        # This would normally connect to database, but we can test the import
        print("✅ Function available for database connection")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 App Citation Integration Test")
    print("=" * 50)
    
    test1 = test_citation_extraction()
    test2 = test_filter_loading()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("🎉 App citation integration working!")
        print("💡 Ready to test in the actual Streamlit app.")
    else:
        print("❌ Some tests failed. Check the app.py integration.")
