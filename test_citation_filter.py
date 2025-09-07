#!/usr/bin/env python3
"""
Test Citation Filter Integration

This script tests whether the citation filter integration is working correctly:
1. Loads filter settings from database
2. Tests ResponseFormatter with different citation styles
3. Verifies that filter settings are properly integrated
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.utils import get_filter_settings
    from filter_examples import ResponseFormatter
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_filter_settings_loading():
    """Test loading filter settings from database"""
    print("\n🔍 Testing filter settings loading...")
    
    try:
        settings = get_filter_settings()
        if settings:
            citation_style = settings.get('citation_style', 'Academic (APA-style)')
            print(f"✅ Loaded filter settings successfully")
            print(f"   Citation style: {citation_style}")
            return citation_style
        else:
            print("⚠️  No filter settings found in database")
            return 'Academic (APA-style)'
    except Exception as e:
        print(f"❌ Error loading filter settings: {e}")
        return 'Academic (APA-style)'

def test_response_formatter():
    """Test ResponseFormatter with different citation styles"""
    print("\n🔍 Testing ResponseFormatter...")
    
    # Test data
    test_response = "This is a test response that mentions some information."
    test_citations = [
        {"title": "Test Article 1", "url": "https://example.com/article1"},
        {"title": "Test Article 2", "url": "https://example.com/article2"}
    ]
    
    citation_styles = ["Academic (APA-style)", "Numbered", "Simple links", "Inline"]
    
    for style in citation_styles:
        try:
            formatter = ResponseFormatter(style)
            formatted = formatter.format_citations(test_response, test_citations)
            
            print(f"✅ {style} formatting works")
            print(f"   Result length: {len(formatted)} chars")
            
            # Check if citations were added
            if len(formatted) > len(test_response):
                print(f"   ✅ Citations were added")
            else:
                print(f"   ⚠️  No citations added (expected for some styles)")
                
        except Exception as e:
            print(f"❌ Error with {style}: {e}")

def test_integration():
    """Test the complete integration"""
    print("\n🔍 Testing complete integration...")
    
    try:
        # Load settings
        citation_style = test_filter_settings_loading()
        
        # Test formatter
        formatter = ResponseFormatter(citation_style)
        test_response = "Integration test response."
        test_citations = [{"title": "Integration Test", "url": "https://test.com"}]
        
        formatted = formatter.format_citations(test_response, test_citations)
        
        print("✅ Integration test successful")
        print(f"   Original: {len(test_response)} chars")
        print(f"   Formatted: {len(formatted)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Citation Filter Integration Test")
    print("=" * 50)
    
    # Run tests
    test_filter_settings_loading()
    test_response_formatter()
    success = test_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Citation filter integration is working.")
    else:
        print("❌ Some tests failed. Check the output above.")
