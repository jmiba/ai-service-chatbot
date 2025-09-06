# Tests

This directory contains all test files for the AI Service Chatbot application.

## Test Files

- **`test_filters.py`** - Comprehensive testing of content filtering functionality including academic integrity, language detection, topic restriction, and citation formatting
- **`test_advanced_parameters.py`** - Testing OpenAI API advanced parameter configurations 
- **`test_implementation_status.py`** - Demonstrates that all filter features are fully implemented and operational
- **`test_reasoning_latency.py`** - Performance testing for reasoning models (GPT-5 series)
- **`test_responses_api_parameters.py`** - Testing Responses API parameter configurations
- **`test_session_id.py`** - Testing session management and conversation tracking
- **`test_streamlit_sim.py`** - Testing Streamlit UI simulation components
- **`test_temperature_detailed.py`** - Testing temperature parameter effects on response generation

## Running Tests

From the root directory, you can run individual tests:

```bash
# Run filter tests
python3 tests/test_filters.py

# Run implementation status validation
python3 tests/test_implementation_status.py

# Run all tests (if using pytest)
pytest tests/
```

## Test Dependencies

Make sure you have the required dependencies installed:
- `streamlit`
- `openai` 
- `psycopg2`
- Any other dependencies from `requirements.txt`
