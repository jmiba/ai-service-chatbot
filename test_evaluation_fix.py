#!/usr/bin/env python3

"""
Test script to verify the evaluation call JSON parsing fix
"""

import json
import re

# Simulate the response format types we might get
test_responses = [
    # Valid JSON
    '{"request_classification": "research_help", "confidence": 0.8, "error_code": 0, "evaluation_notes": "Good response"}',
    
    # JSON wrapped in text
    'Here is the evaluation:\n{"request_classification": "research_help", "confidence": 0.8, "error_code": 0, "evaluation_notes": "Good response"}\nEnd of evaluation.',
    
    # Invalid JSON (missing quote)
    '{"request_classification": research_help", "confidence": 0.8, "error_code": 0, "evaluation_notes": "Good response"}',
    
    # JSON with extra comma (the specific error from the user)
    '{"request_classification": "research_help", "confidence": 0.8, "error_code": 0, "evaluation_notes": "Good response",}',
]

def parse_evaluation_response(payload_text):
    """Test the improved JSON parsing logic"""
    try:
        # First attempt: direct parsing
        payload = json.loads(payload_text)
        print(f"✅ Direct parsing successful: {payload}")
        return payload
    except json.JSONDecodeError as json_err:
        print(f"❌ Direct parsing failed: {json_err}")
        
        # Try to extract JSON from response if it's wrapped in text
        json_match = re.search(r'\{.*\}', payload_text, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group())
                print(f"✅ Extraction parsing successful: {payload}")
                return payload
            except json.JSONDecodeError:
                print(f"❌ Extraction parsing also failed: {json_match.group()}")
                raise json_err
        else:
            print(f"❌ No JSON found in response: {payload_text}")
            raise json_err

def test_all_responses():
    """Test all response formats"""
    print("Testing JSON parsing improvements...")
    print("=" * 50)
    
    for i, response in enumerate(test_responses, 1):
        print(f"\nTest {i}: {response[:50]}...")
        try:
            result = parse_evaluation_response(response)
            print(f"✅ SUCCESS: {result}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            # This would use default values in the real code
            print("Would continue with default values in production")

if __name__ == "__main__":
    test_all_responses()
