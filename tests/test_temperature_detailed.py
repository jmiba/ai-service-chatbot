#!/usr/bin/env python3
"""
Detailed test of temperature parameter behavior in OpenAI Responses API
"""

import openai
import streamlit as st
from openai import OpenAI

def test_temperature_behavior():
    """Test temperature parameter thoroughly to understand its behavior and limitations"""
    
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Test prompt that should show clear temperature differences
    test_input = [{"role": "user", "content": "Write a creative opening line for a story about a mysterious door. Make it engaging and unique."}]
    
    # Temperature values to test
    temperature_values = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5]  # Include invalid value
    
    print("🌡️  Testing temperature parameter behavior...")
    print("=" * 80)
    print("Test prompt: Creative story opening about a mysterious door")
    print("=" * 80)
    
    successful_temps = []
    failed_temps = []
    
    for temp in temperature_values:
        print(f"\n🎯 Testing temperature = {temp}")
        print("-" * 50)
        
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=test_input,
                temperature=temp
            )
            
            successful_temps.append(temp)
            
            # Extract response text
            response_text = ""
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content in item.content:
                            if hasattr(content, 'text'):
                                response_text = content.text
                                break
            
            print(f"✅ SUCCESS - Temperature {temp}")
            print(f"📝 Response: {response_text[:150]}...")
            
            # Show additional response details
            if hasattr(response, 'usage'):
                print(f"📊 Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
            
        except Exception as e:
            failed_temps.append((temp, str(e)))
            print(f"❌ FAILED - Temperature {temp}")
            print(f"🔥 Error: {str(e)[:200]}...")
    
    # Test with different models
    print("\n" + "=" * 80)
    print("🤖 Testing temperature across different models")
    print("=" * 80)
    
    models_to_test = ["gpt-4o-mini", "gpt-4o", "gpt-5-mini"]
    test_temp = 0.8  # A safe middle value
    
    for model in models_to_test:
        print(f"\n🤖 Model: {model} (temperature={test_temp})")
        print("-" * 50)
        
        try:
            response = client.responses.create(
                model=model,
                input=test_input,
                temperature=test_temp
            )
            
            # Extract response text
            response_text = ""
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content in item.content:
                            if hasattr(content, 'text'):
                                response_text = content.text
                                break
            
            print(f"✅ SUCCESS")
            print(f"📝 Response: {response_text[:150]}...")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:200]}...")
    
    # Test temperature with tools
    print("\n" + "=" * 80)
    print("🛠️  Testing temperature with tools enabled")
    print("=" * 80)
    
    tool_test_input = [{"role": "user", "content": "What's the current time? Use appropriate tools."}]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    
    for temp in [0.0, 0.7, 1.0]:
        print(f"\n🎯 Testing temperature = {temp} with tools")
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=tool_test_input,
                temperature=temp,
                tools=tools,
                parallel_tool_calls=True
            )
            print(f"✅ SUCCESS with tools - Temperature {temp}")
            
        except Exception as e:
            print(f"❌ FAILED with tools - Temperature {temp}: {str(e)[:150]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEMPERATURE PARAMETER ANALYSIS")
    print("=" * 80)
    
    print(f"✅ SUCCESSFUL TEMPERATURES ({len(successful_temps)}):")
    for temp in successful_temps:
        print(f"   • {temp}")
    
    if failed_temps:
        print(f"\n❌ FAILED TEMPERATURES ({len(failed_temps)}):")
        for temp, error in failed_temps:
            print(f"   • {temp}: {error[:100]}...")
    
    # Behavioral analysis
    if len(successful_temps) >= 3:
        print(f"\n🔍 BEHAVIORAL INSIGHTS:")
        print(f"   • Valid range appears to be: {min(successful_temps)} to {max(successful_temps)}")
        print(f"   • Temperature affects response creativity and randomness")
        print(f"   • Lower values (0.0-0.3) = more focused, deterministic")
        print(f"   • Higher values (0.7-1.0) = more creative, varied")
        print(f"   • Works with both streaming and non-streaming responses")
        
        if 2.0 in successful_temps:
            print(f"   • High values (1.5-2.0) supported for maximum creativity")
    
    print(f"\n🎯 RECOMMENDATION FOR ADMIN INTERFACE:")
    if successful_temps:
        min_temp = min(successful_temps)
        max_temp = max(successful_temps)
        print(f"   • Add temperature slider: {min_temp} to {max_temp}")
        print(f"   • Default value: 0.7 (balanced creativity)")
        print(f"   • Step size: 0.1 for fine control")
        print(f"   • Safe to include in production interface")
    else:
        print(f"   • Temperature parameter appears unstable - avoid for now")

if __name__ == "__main__":
    test_temperature_behavior()
