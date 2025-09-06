# Tests

This directory contains all test files for the AI Service Chatbot application.

## Quick Start

**Run all tests quickly (no slow API calls):**
```bash
python3 tests/run_quick_tests.py
```

**Run individual fast tests:**
```bash
python3 tests/test_filters.py
python3 tests/test_implementation_status.py
python3 tests/test_session_id.py
```

**Run slow API tests (only when needed):**
```bash
python3 tests/test_reasoning_latency.py --run-api-tests
python3 tests/test_temperature_detailed.py --run-api-tests
```

## Test Files

### 🚀 Fast Tests (no API calls)
- **`test_filters.py`** - Comprehensive testing of content filtering functionality including academic integrity, language detection, topic restriction, and citation formatting
- **`test_implementation_status.py`** - Demonstrates that all filter features are fully implemented and operational
- **`test_session_id.py`** - Testing session management and conversation tracking
- **`test_streamlit_sim.py`** - Testing Streamlit UI simulation components

### ⏳ Slow Tests (make API calls)
- **`test_reasoning_latency.py`** - Performance testing for reasoning models (GPT-5 series) - can take 30+ seconds
- **`test_advanced_parameters.py`** - Testing OpenAI API advanced parameter configurations 
- **`test_responses_api_parameters.py`** - Testing Responses API parameter configurations
- **`test_temperature_detailed.py`** - Testing temperature parameter effects on response generation

### 🏃‍♂️ Test Runner
- **`run_quick_tests.py`** - Runs all fast tests in sequence, skips slow API calls

## Running Tests

From the root directory:

```bash
# Quick test suite (recommended for development)
python3 tests/run_quick_tests.py

# Individual fast tests
python3 tests/test_filters.py
python3 tests/test_implementation_status.py

# Slow API tests (only when testing API integration)
python3 tests/test_reasoning_latency.py --run-api-tests

# All tests with pytest (if installed)
pytest tests/
```

## Test Dependencies

Make sure you have the required dependencies installed:
- `streamlit`
- `openai` 
- `psycopg2`
- Any other dependencies from `requirements.txt`
