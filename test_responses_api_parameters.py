#!/usr/bin/env python3
"""
Test script to discover which parameters are actually supported by OpenAI Responses API
"""

import openai
import streamlit as st
from openai import OpenAI

def test_responses_api_parameters():
    """Test various parameters with the Responses API to see what's supported"""
    
    # Initialize client
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        return
    
    # Base test message
    test_input = [{"role": "user", "content": "Say 'Hello' in exactly 3 words."}]
    
    # Parameters to test (from the response dump)
    test_parameters = {
        'temperature': 0.8,
        'top_p': 0.9,
        'max_output_tokens': 50,
        'max_tool_calls': 1,
        'tool_choice': 'auto',
        'store': True,
        'top_logprobs': 1,
        'reasoning': {'effort': 'low'},
        'text': {'verbosity': 'low'},
        'service_tier': 'default'
    }
    
    # Test each parameter individually
    supported_params = {}
    unsupported_params = {}
    
    print("🧪 Testing OpenAI Responses API parameter support...")
    print("=" * 60)
    
    # Test base call first
    try:
        base_response = client.responses.create(
            model="gpt-4o-mini",
            input=test_input
        )
        print("✅ Base API call successful")
        print(f"   Response ID: {base_response.id}")
        print()
    except Exception as e:
        print(f"❌ Base API call failed: {e}")
        return
    
    # Test each parameter individually
    for param_name, param_value in test_parameters.items():
        try:
            kwargs = {
                'model': 'gpt-4o-mini',
                'input': test_input,
                param_name: param_value
            }
            
            print(f"Testing {param_name}: {param_value}...")
            
            response = client.responses.create(**kwargs)
            supported_params[param_name] = param_value
            print(f"✅ {param_name} - SUPPORTED")
            
        except Exception as e:
            unsupported_params[param_name] = str(e)
            print(f"❌ {param_name} - NOT SUPPORTED: {str(e)[:100]}")
        
        print()
    
    # Test combinations of supported parameters
    if supported_params:
        print("🔄 Testing combination of supported parameters...")
        try:
            combo_kwargs = {
                'model': 'gpt-4o-mini',
                'input': test_input,
                **supported_params
            }
            response = client.responses.create(**combo_kwargs)
            print("✅ All supported parameters work together!")
            
            # Print detailed response info
            print(f"   Model used: {getattr(response, 'model', 'N/A')}")
            if hasattr(response, 'usage'):
                print(f"   Input tokens: {response.usage.input_tokens}")
                print(f"   Output tokens: {response.usage.output_tokens}")
            
        except Exception as e:
            print(f"❌ Combination failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    print(f"✅ SUPPORTED PARAMETERS ({len(supported_params)}):")
    for param, value in supported_params.items():
        print(f"   • {param}: {value}")
    
    print(f"\n❌ UNSUPPORTED PARAMETERS ({len(unsupported_params)}):")
    for param, error in unsupported_params.items():
        print(f"   • {param}: {error[:80]}...")
    
    print(f"\n🎯 RECOMMENDATION:")
    if supported_params:
        print(f"   Add these {len(supported_params)} parameters to the admin interface")
        print("   They can be safely used with the Responses API")
    else:
        print("   Keep the current minimal parameter set")
        print("   Most advanced parameters are not supported")

if __name__ == "__main__":
    test_responses_api_parameters()
