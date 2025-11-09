#!/usr/bin/env python3

"""
Test script to verify session ID functionality
"""

import uuid
from utils.utils import get_connection
import psycopg2
from datetime import datetime

def test_log_interaction(user_input, assistant_response, session_id=None, citation_json=None, citation_count=0, confidence=0.0, request_type=None, request_classification=None, evaluation_notes=None):
    """Test version of log_interaction function"""
    print(f"🔍 Testing log_interaction with:")
    print(f"  - session_id: {session_id}")
    print(f"  - user_input: {user_input[:50]}...")
    print(f"  - assistant_response: {assistant_response[:50]}...")
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO log_table (timestamp, session_id, user_input, assistant_response, request_type, citation_count, citations, confidence, request_classification, evaluation_notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (datetime.now(), session_id, user_input, assistant_response, request_type, citation_count, citation_json, confidence, request_classification, evaluation_notes))
                conn.commit()
                print("✅ Successfully logged interaction")
    except psycopg2.Error as e:
        print(f"❌ DB logging error: {e}")

def verify_recent_entries():
    """Check the most recent entries in the database"""
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
    print("🧪 Testing Session ID functionality")
    
    # Generate a test session ID
    test_session_id = str(uuid.uuid4())
    print(f"Generated test session ID: {test_session_id}")
    
    # Test logging with session ID
    test_log_interaction(
        user_input="Test question about library hours",
        assistant_response="The library is open from 8am to 10pm on weekdays",
        session_id=test_session_id,
        citation_count=1,
        confidence=0.9,
        request_type="E00",
        request_classification="library_hours",
        evaluation_notes="Test evaluation"
    )
    
    # Verify the entry was created
    verify_recent_entries()
    
    print("\n🎯 Test completed!")
