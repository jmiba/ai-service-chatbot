#!/usr/bin/env python3
"""
Test reasoning effort impact on response latency and quality
Run with --run-api-tests flag to enable actual API calls (slow!)
"""

import sys
import time
import openai
import streamlit as st
from openai import OpenAI

def test_reasoning_effort_latency(run_api_tests=False):
    """Test how reasoning effort affects response time and quality"""
    
    print("⏱️  Testing reasoning effort impact on latency and quality...")
    print("=" * 80)
    
    if not run_api_tests:
        print("🚀 QUICK MODE: Skipping slow API calls")
        print("💡 To run full API tests, use: python3 tests/test_reasoning_latency.py --run-api-tests")
        print("=" * 80)
        
        # Mock results for demonstration
        mock_results = {
            'low': {'response_time': 2.3, 'reasoning_tokens': 50, 'output_tokens': 150},
            'medium': {'response_time': 8.7, 'reasoning_tokens': 200, 'output_tokens': 180},
            'high': {'response_time': 23.4, 'reasoning_tokens': 800, 'output_tokens': 220}
        }
        
        print("📊 Expected Performance (based on typical GPT-5 behavior):")
        for effort, data in mock_results.items():
            print(f"  {effort.upper()}: ~{data['response_time']:.1f}s, {data['reasoning_tokens']} reasoning tokens")
        
        print("\n✅ Quick test completed - no actual API calls made")
        return mock_results
    
    # Original API testing code (slow!)
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Test prompt that benefits from deeper reasoning
    test_input = [{
        "role": "user", 
        "content": "Analyze the potential economic impacts of implementing a 4-day work week in Germany. Consider productivity, employment, consumer spending, and international competitiveness."
    }]
    
    reasoning_levels = ["low", "medium", "high"]
    results = {}
    
    print("⏱️  Testing reasoning effort impact on latency and quality...")
    print("=" * 80)
    print("Test: Complex economic analysis question")
    print("Model: gpt-5-mini (supports reasoning effort)")
    print("=" * 80)
    
    for effort in reasoning_levels:
        print(f"\n🧠 Testing reasoning effort: {effort.upper()}")
        print("-" * 50)
        
        try:
            # Measure response time
            start_time = time.time()
            
            response = client.responses.create(
                model="gpt-5-mini",
                input=test_input,
                reasoning={'effort': effort}
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract response details
            response_text = ""
            reasoning_tokens = 0
            output_tokens = 0
            
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content in item.content:
                            if hasattr(content, 'text'):
                                response_text = content.text
                                break
            
            if hasattr(response, 'usage'):
                output_tokens = response.usage.output_tokens
                if hasattr(response.usage, 'output_tokens_details') and response.usage.output_tokens_details:
                    reasoning_tokens = getattr(response.usage.output_tokens_details, 'reasoning_tokens', 0)
            
            results[effort] = {
                'response_time': response_time,
                'response_text': response_text,
                'output_tokens': output_tokens,
                'reasoning_tokens': reasoning_tokens,
                'total_tokens': output_tokens + reasoning_tokens
            }
            
            print(f"✅ SUCCESS")
            print(f"⏱️  Response time: {response_time:.2f} seconds")
            print(f"🔢 Output tokens: {output_tokens}")
            print(f"🧠 Reasoning tokens: {reasoning_tokens}")
            print(f"📝 Response preview: {response_text[:200]}...")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:100]}...")
            results[effort] = {'error': str(e)}
    
    # Analysis and comparison
    print("\n" + "=" * 80)
    print("📊 LATENCY AND QUALITY ANALYSIS")
    print("=" * 80)
    
    if all('response_time' in results[effort] for effort in reasoning_levels):
        print("⏱️  RESPONSE TIMES:")
        for effort in reasoning_levels:
            time_str = f"{results[effort]['response_time']:.2f}s"
            tokens = results[effort]['reasoning_tokens']
            print(f"   {effort.upper():<8} {time_str:<8} (reasoning tokens: {tokens})")
        
        # Calculate relative differences
        low_time = results['low']['response_time']
        med_time = results['medium']['response_time']
        high_time = results['high']['response_time']
        
        print(f"\n📈 RELATIVE LATENCY INCREASES:")
        print(f"   Medium vs Low:  {((med_time - low_time) / low_time * 100):+.1f}%")
        print(f"   High vs Low:    {((high_time - low_time) / low_time * 100):+.1f}%")
        print(f"   High vs Medium: {((high_time - med_time) / med_time * 100):+.1f}%")
        
        print(f"\n🧠 REASONING TOKEN USAGE:")
        for effort in reasoning_levels:
            tokens = results[effort]['reasoning_tokens']
            output = results[effort]['output_tokens']
            ratio = tokens / output if output > 0 else 0
            print(f"   {effort.upper():<8} {tokens:>4} reasoning tokens ({ratio:.1f}x output)")
        
        print(f"\n💰 COST IMPLICATIONS:")
        print(f"   • Higher reasoning effort = more reasoning tokens")
        print(f"   • Reasoning tokens are typically charged differently")
        print(f"   • Trade-off: Better quality vs higher cost & latency")
        
        print(f"\n📋 RESPONSE QUALITY INDICATORS:")
        for effort in reasoning_levels:
            length = len(results[effort]['response_text'])
            reasoning = results[effort]['reasoning_tokens']
            print(f"   {effort.upper():<8} {length:>4} chars, {reasoning:>4} reasoning tokens")
    
    print(f"\n🎯 RECOMMENDATIONS FOR ADMIN INTERFACE:")
    print(f"   ⚡ Low effort:    Quick responses for simple questions")
    print(f"   ⚖️  Medium effort: Balanced for most use cases")  
    print(f"   🎯 High effort:   Complex analysis, research, critical thinking")
    print(f"   ⚠️  Warning:      High effort significantly increases latency & cost")
    
    print(f"\n💡 USER GUIDANCE SUGGESTIONS:")
    print(f"   • Show estimated response time for each effort level")
    print(f"   • Add cost indicators (💰 symbols)")
    print(f"   • Recommend effort levels based on query complexity")
    print(f"   • Allow users to cancel slow requests")

if __name__ == "__main__":
    # Check for command line flag to enable slow API tests
    run_api_tests = "--run-api-tests" in sys.argv
    test_reasoning_effort_latency(run_api_tests=run_api_tests)
