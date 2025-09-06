#!/usr/bin/env python3
"""
Test reasoning and text parameters with different models to see which ones support them
"""

import openai
import streamlit as st
from openai import OpenAI

def test_advanced_parameters():
    """Test reasoning and text parameters with different models"""
    
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    test_input = [{"role": "user", "content": "Explain photosynthesis in 2 sentences."}]
    
    # Models to test
    models_to_test = ["gpt-4o-mini", "gpt-4o", "gpt-5-mini"]
    
    # Advanced parameters to test
    advanced_params = {
        'reasoning_effort_low': {'reasoning': {'effort': 'low'}},
        'reasoning_effort_medium': {'reasoning': {'effort': 'medium'}},
        'reasoning_effort_high': {'reasoning': {'effort': 'high'}},
        'text_verbosity_low': {'text': {'verbosity': 'low'}},
        'text_verbosity_medium': {'text': {'verbosity': 'medium'}},
        'text_verbosity_high': {'text': {'verbosity': 'high'}},
    }
    
    print("🧪 Testing advanced parameters across different models...")
    print("=" * 70)
    
    results = {}
    
    for model in models_to_test:
        print(f"\n🤖 Testing model: {model}")
        print("-" * 50)
        
        results[model] = {}
        
        for param_name, param_dict in advanced_params.items():
            try:
                kwargs = {
                    'model': model,
                    'input': test_input,
                    **param_dict
                }
                
                response = client.responses.create(**kwargs)
                results[model][param_name] = "✅ SUPPORTED"
                print(f"✅ {param_name}")
                
                # For successful calls, show some response details
                if hasattr(response, 'output') and response.output:
                    output_text = ""
                    for item in response.output:
                        if hasattr(item, 'content') and item.content:
                            for content in item.content:
                                if hasattr(content, 'text'):
                                    output_text = content.text[:100] + "..."
                                    break
                    if output_text:
                        print(f"   Response: {output_text}")
                
            except Exception as e:
                error_msg = str(e)
                results[model][param_name] = f"❌ {error_msg[:60]}..."
                print(f"❌ {param_name}: {error_msg[:60]}...")
    
    # Summary table
    print("\n" + "=" * 70)
    print("📊 COMPATIBILITY MATRIX")
    print("=" * 70)
    
    # Header
    print(f"{'Parameter':<25} {'gpt-4o-mini':<15} {'gpt-4o':<15} {'gpt-5-mini':<15}")
    print("-" * 70)
    
    for param_name in advanced_params.keys():
        row = f"{param_name:<25}"
        for model in models_to_test:
            status = results.get(model, {}).get(param_name, "❓ Not tested")
            symbol = "✅" if "SUPPORTED" in status else "❌"
            row += f" {symbol:<15}"
        print(row)
    
    # Recommendations
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATIONS FOR ADMIN INTERFACE")
    print("=" * 70)
    
    # Find universally supported parameters
    universal_params = []
    model_specific_params = []
    
    for param_name in advanced_params.keys():
        supported_count = sum(1 for model in models_to_test 
                            if "SUPPORTED" in results.get(model, {}).get(param_name, ""))
        
        if supported_count == len(models_to_test):
            universal_params.append(param_name)
        elif supported_count > 0:
            model_specific_params.append((param_name, supported_count))
    
    if universal_params:
        print("✅ SAFE TO ADD (supported by all models):")
        for param in universal_params:
            print(f"   • {param}")
    
    if model_specific_params:
        print("\n⚠️  MODEL-DEPENDENT (consider conditional support):")
        for param, count in model_specific_params:
            print(f"   • {param} (supported by {count}/{len(models_to_test)} models)")
    
    # Final parameter list recommendation
    print("\n🔧 SUGGESTED ADMIN INTERFACE PARAMETERS:")
    basic_params = ["model", "parallel_tool_calls", "temperature", "top_p", "max_output_tokens", "tool_choice"]
    print("   Basic parameters (always include):")
    for param in basic_params:
        print(f"     • {param}")
    
    if universal_params:
        print("   Advanced parameters (safe to add):")
        for param in universal_params:
            print(f"     • {param}")

if __name__ == "__main__":
    test_advanced_parameters()
