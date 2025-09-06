#!/usr/bin/env python3
"""
Database Schema Diagnostic Tool
Checks if session_id column exists and fixes it if needed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import check_log_table_schema, force_add_session_id_column, get_connection

def main():
    print("🔍 Checking database schema...")
    
    try:
        # Check current schema
        columns = check_log_table_schema()
        print(f"\n📋 Current log_table columns:")
        for col_name, data_type, is_nullable in columns:
            print(f"  - {col_name} ({data_type}) {'NULL' if is_nullable == 'YES' else 'NOT NULL'}")
        
        # Check if session_id exists
        session_id_exists = any(col[0] == 'session_id' for col in columns)
        
        if session_id_exists:
            print("\n✅ session_id column found in log_table!")
            
            # Test if we can query with session_id
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM log_table WHERE session_id IS NOT NULL;")
            count_with_session = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM log_table;")
            total_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            print(f"📊 Log entries: {total_count} total, {count_with_session} with session_id")
            
        else:
            print("\n❌ session_id column NOT found in log_table!")
            print("🔧 Attempting to add session_id column...")
            
            if force_add_session_id_column():
                print("✅ session_id column added successfully!")
                
                # Re-check schema
                columns = check_log_table_schema()
                print(f"\n📋 Updated log_table columns:")
                for col_name, data_type, is_nullable in columns:
                    print(f"  - {col_name} ({data_type}) {'NULL' if is_nullable == 'YES' else 'NOT NULL'}")
            else:
                print("❌ Failed to add session_id column. Check error messages above.")
                return 1
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("\n🎉 Database schema check complete!")
    return 0

if __name__ == "__main__":
    exit(main())
