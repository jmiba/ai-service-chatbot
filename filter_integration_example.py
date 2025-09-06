"""
Filter Integration Example for app.py

This demonstrates how to integrate the comprehensive filtering system 
into your main chatbot application.
"""

import streamlit as st
import json
from filter_examples import apply_comprehensive_filters
from utils.utils import get_filter_settings

def apply_filters_to_response(user_input, ai_response, metadata):
    """
    Apply all configured filters to a chatbot response.
    
    Args:
        user_input (str): The user's original question
        ai_response (str): The AI's generated response
        metadata (dict): Response metadata (confidence, citations, etc.)
    
    Returns:
        dict: Filter results with filtered response and decisions
    """
    
    try:
        # Load current filter settings from database
        filter_settings = get_filter_settings()
        
        # Apply comprehensive filters
        filter_result = apply_comprehensive_filters(
            user_input, ai_response, metadata, filter_settings
        )
        
        return filter_result
        
    except Exception as e:
        st.error(f"Filter error: {e}")
        # Return original response if filters fail
        return {
            'original_response': ai_response,
            'filtered_response': ai_response,
            'should_block': False,
            'warnings': [f"Filter system error: {e}"],
            'modifications': [],
            'user_guidance': None
        }

def handle_blocked_response(filter_result, user_input):
    """
    Handle responses that were blocked by filters.
    Provide appropriate guidance to the user.
    """
    
    warnings = filter_result.get('warnings', [])
    
    # Academic integrity guidance
    if any('academic integrity' in warning.lower() for warning in warnings):
        return """
        🎓 **Lernhilfe statt Lösung**
        
        Ich helfe gerne beim Verstehen von Konzepten und Methoden, aber ich löse keine Aufgaben direkt. 
        Das würde Ihrem Lernprozess nicht helfen!
        
        **Stattdessen kann ich:**
        - Die relevanten Konzepte erklären
        - Lösungsansätze aufzeigen
        - Ähnliche Beispiele durchgehen
        - Auf Lernressourcen verweisen
        
        Möchten Sie, dass ich Ihnen beim Verständnis der zugrundeliegenden Konzepte helfe?
        """
    
    # Low confidence guidance  
    elif any('confidence' in warning.lower() for warning in warnings):
        return """
        🤔 **Unsichere Antwort erkannt**
        
        Ich bin mir bei dieser Antwort nicht sicher genug. Das könnte bedeuten:
        - Die Information ist nicht in meiner Wissensbasis verfügbar
        - Die Frage ist zu spezifisch oder mehrdeutig
        - Es werden aktuelle Informationen benötigt
        
        **Empfehlung:**
        - Versuchen Sie eine spezifischere Frage
        - Kontaktieren Sie direkt die zuständige Stelle
        - Prüfen Sie die offiziellen Universitätswebseiten
        """
    
    # Topic restriction guidance
    elif any('topic' in warning.lower() for warning in warnings):
        return """
        📚 **Außerhalb des Universitätskontexts**
        
        Mein Fokus liegt auf universitätsbezogenen Themen wie:
        - Studium und Kurse
        - Forschung und akademische Ressourcen  
        - Campusleben und -services
        - Universitätspolitik und -verfahren
        
        Für andere Themen wenden Sie sich bitte an entsprechende Fachstellen oder Beratungsdienste.
        """
    
    # Generic blocked response
    else:
        return """
        ⚠️ **Antwort kann nicht bereitgestellt werden**
        
        Die generierte Antwort entspricht nicht unseren Qualitätsstandards.
        
        **Bitte versuchen Sie:**
        - Eine spezifischere Frage zu stellen
        - Den Kontext zu erweitern
        - Bei dringenden Angelegenheiten direkt die Verwaltung zu kontaktieren
        """

def display_filter_warnings(warnings):
    """Display filter warnings in the UI for admin users"""
    
    if warnings and st.session_state.get('authenticated', False):
        with st.expander("🔧 Filter Warnings (Admin Only)"):
            for warning in warnings:
                st.warning(f"⚠️ {warning}")

def log_filter_results(filter_result, session_id):
    """Log filter results for monitoring and improvement"""
    
    try:
        from utils.utils import get_connection
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Log filter activity (you might want to create a separate filter_logs table)
        cursor.execute("""
            INSERT INTO log_table (
                session_id, user_input, assistant_response, 
                evaluation_notes, timestamp
            ) VALUES (%s, %s, %s, %s, NOW())
        """, (
            session_id,
            "FILTER_LOG", 
            json.dumps({
                'should_block': filter_result['should_block'],
                'warnings': filter_result['warnings'],
                'modifications': filter_result['modifications']
            }),
            f"Filter activity - Blocked: {filter_result['should_block']}"
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        st.error(f"Logging error: {e}")

# Example integration into your main chat function
def enhanced_chat_with_filters(user_input, session_id):
    """
    Enhanced chat function that includes comprehensive filtering.
    This would replace or enhance your existing chat logic in app.py
    """
    
    # Your existing OpenAI API call here (simplified)
    # ai_response, metadata = call_openai_api(user_input)
    
    # For demonstration, mock response
    ai_response = "This is a sample AI response..."
    metadata = {
        'confidence': 0.85,
        'citation_count': 1,
        'citations': [{'title': 'University Guide', 'url': 'https://example.com'}]
    }
    
    # Apply filters
    filter_result = apply_filters_to_response(user_input, ai_response, metadata)
    
    # Log filter results
    log_filter_results(filter_result, session_id)
    
    # Handle blocked responses
    if filter_result['should_block']:
        blocked_response = handle_blocked_response(filter_result, user_input)
        st.warning("Response was filtered for quality/policy reasons")
        st.markdown(blocked_response)
        
        # Show warnings to admins
        display_filter_warnings(filter_result['warnings'])
        return None
    
    # Display filtered response
    st.markdown(filter_result['filtered_response'])
    
    # Show filter modifications to admins
    if filter_result['modifications'] and st.session_state.get('authenticated', False):
        with st.expander("🔧 Applied Modifications (Admin Only)"):
            for mod in filter_result['modifications']:
                st.info(f"✅ {mod}")
    
    # Show warnings to admins
    display_filter_warnings(filter_result['warnings'])
    
    return filter_result['filtered_response']

# Example of how to modify your existing app.py
INTEGRATION_EXAMPLE = '''
# In your app.py, modify the chat processing section:

# Before (existing code):
if st.button("Send", type="primary"):
    if user_input:
        # Your existing OpenAI API call
        response = call_openai_api(user_input)
        st.markdown(response)

# After (with filters):
if st.button("Send", type="primary"):
    if user_input:
        # Your existing OpenAI API call  
        ai_response, metadata = call_openai_api_with_metadata(user_input)
        
        # Apply filters
        filter_result = apply_filters_to_response(user_input, ai_response, metadata)
        
        if filter_result['should_block']:
            # Show blocked response guidance
            blocked_response = handle_blocked_response(filter_result, user_input)
            st.warning("Response filtered for quality/policy")
            st.markdown(blocked_response)
        else:
            # Show filtered response
            st.markdown(filter_result['filtered_response'])
            
            # Show admin info if authenticated
            if st.session_state.get('authenticated'):
                display_filter_warnings(filter_result['warnings'])
'''

if __name__ == "__main__":
    print("🔧 Filter Integration Guide")
    print("=" * 50)
    print("This file shows how to integrate filters into your main app.py")
    print("\n📋 Integration Steps:")
    print("1. Import filter functions into app.py")
    print("2. Call apply_filters_to_response() after AI response")  
    print("3. Handle blocked responses with appropriate guidance")
    print("4. Display filtered response to user")
    print("5. Log filter results for monitoring")
    
    print(f"\n📝 Example Code:")
    print(INTEGRATION_EXAMPLE)
    
    print(f"\n🎯 Benefits After Integration:")
    print("✅ Automatic quality control")
    print("✅ Academic integrity protection") 
    print("✅ Consistent citation formatting")
    print("✅ Topic-appropriate responses")
    print("✅ User-adaptive language")
    print("✅ Admin monitoring capabilities")
