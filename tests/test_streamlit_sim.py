#!/usr/bin/env python3

"""
Test script to simulate Streamlit session behavior and verify session ID flow
"""

import uuid
import json
from datetime import datetime
from utils.utils import get_connection
import psycopg2

# Simulate Streamlit session state
class MockSessionState:
    def __init__(self):
        self._state = {}
    
    def __contains__(self, key):
        return key in self._state
    
    def __getattr__(self, key):
        return self._state.get(key)
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._state[key] = value

# Create mock session state
st_session_state = MockSessionState()

def log_interaction(user_input, assistant_response, session_id=None, citation_json=None, citation_count=0, confidence=0.0, request_type=None, request_classification=None, evaluation_notes=None):
    # Debug: Print session_id to console for troubleshooting
    print(f"🔍 log_interaction called with session_id: {session_id} (type: {type(session_id)})")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO log_table (timestamp, session_id, user_input, assistant_response, request_type, citation_count, citations, confidence, request_classification, evaluation_notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (datetime.now(), session_id, user_input, assistant_response, request_type, citation_count, citation_json, confidence, request_classification, evaluation_notes))
                conn.commit()
                print(f"✅ Successfully logged interaction with session_id: {session_id}")
    except psycopg2.Error as e:
        print(f"❌ DB logging error: {e}")

def simulate_streamlit_session():
    """Simulate how Streamlit handles session state"""
    print("🧪 Simulating Streamlit session behavior")
    
    # Initialize unique session ID for conversation tracking (like in app.py)
    if "session_id" not in st_session_state:
        st_session_state.session_id = str(uuid.uuid4())
        print(f"🆕 Generated new session_id: {st_session_state.session_id}")
    else:
        print(f"♻️ Using existing session_id: {st_session_state.session_id}")
    
    # Simulate user interaction
    user_input = "Test question from simulated session"
    cleaned = "Test response from simulated session"
    citation_map = {}
    confidence = 0.85
    request_type = "E00"
    request_classification = "test"
    evaluation_notes = "Simulated evaluation"
    
    # Debug: Check session ID before logging (like in app.py)
    print(f"🔎 About to log interaction with session_id: {st_session_state.session_id}")
    
    # Call log_interaction exactly like in app.py
    log_interaction(
        user_input=user_input,
        assistant_response=cleaned,
        session_id=st_session_state.session_id,
        citation_json=json.dumps(citation_map, ensure_ascii=False) if citation_map else None,
        citation_count=len(citation_map),
        confidence=confidence,
        request_type=request_type,
        request_classification=request_classification,
        evaluation_notes=evaluation_notes
    )

def check_recent_entries():
    """Check the most recent entries"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, timestamp, session_id, user_input
                    FROM log_table 
                    ORDER BY timestamp DESC 
                    LIMIT 3
                """)
                rows = cur.fetchall()
                
                print("\n📋 Recent log entries:")
                for row in rows:
                    id_val, timestamp, session_id, user_input = row
                    print(f"  ID: {id_val}, Session: {session_id}, Input: {user_input[:30]}...")
                    
    except psycopg2.Error as e:
        print(f"❌ Error checking entries: {e}")

if __name__ == "__main__":
    print("🧪 Testing Session ID flow simulation")
    
    # Simulate first run
    simulate_streamlit_session()
    
    # Simulate second run (session should persist)
    print("\n--- Simulating second interaction in same session ---")
    simulate_streamlit_session()
    
    # Check results
    check_recent_entries()
    
    print("\n🎯 Simulation completed!")
