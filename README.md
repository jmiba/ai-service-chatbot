# Viadrina Library Assistant 🤖📚

An intelligent AI-powered chatbot built with Streamlit and OpenAI that helps users with library-related questions. The assistant provides contextual answers with source citations using RAG (Retrieval-Augmented Generation) and includes comprehensive logging and evaluation capabilities.

![Viadrina Library Assistant](assets/viadrina-logo.png)

## 🌐 Live Demo

**Try the live application:** [viadrina.streamlit.app](https://viadrina.streamlit.app)

Experience the Viadrina Library Assistant in action! The live demo includes the full chat interface with knowledge base search, web search capabilities, and real-time response streaming.

## ✨ Features

### 🎯 Core Functionality
- **Intelligent Q&A**: Natural language processing for library-related queries in any language
- **Source Citations**: Automatic citation generation with hover tooltips showing document summaries
- **Multi-language Support**: Ask questions in German, English, or any language
- **Real-time Streaming**: Live response generation with human-readable status indicators
- **Enhanced UX**: Visual spinner with contextual status messages ("Searching the web…", "Searching my knowledge base…", "Generating answer…")
- **Conversation Context**: Maintains conversation history for better context understanding
- **Session Tracking**: UUID-based session identification for conversation analytics

### 📊 Advanced Analytics & Management
- **Knowledge Base Management**: Enhanced web scraping with LLM-powered content analysis
- **Terminal-Style Logging**: Real-time progress updates in admin interfaces with live output capture
- **Individual Save Controls**: Granular URL configuration management with individual save/delete buttons
- **Response Evaluation**: Automatic classification, confidence scoring, and quality assessment
- **Session Analytics**: Conversation grouping and flow analysis by session ID
- **Comprehensive Logging**: Database logging with evaluation metrics, session tracking, and detailed analytics
- **Citation Management**: Smart citation extraction with database-backed metadata and source linking

### 🔧 Technical Features
- **RAG Implementation**: Vector search with OpenAI's file search capabilities and 5-minute caching
- **Web Search Integration**: External web search with contextual status updates and retrieval filters
- **Database Integration**: PostgreSQL with automatic schema migration and diagnostic tools
- **Enhanced Streaming**: Human-readable progress indicators with visual spinner and tool detection
- **Functional Vector Sync**: Working vector store synchronization with real-time feedback
- **Error Handling**: Graceful degradation when services are unavailable
- **Cloud-Ready**: Full deployment support for Streamlit Cloud with external databases

### 🆕 Recent Enhancements
- **LLM Output Visibility**: Real-time display of LLM analysis results during content scraping
- **Vector Store Sync**: Functional synchronization button with progress tracking
- **Session-Based Conversation Tracking**: Complete conversation analytics infrastructure
- **Terminal Output Capture**: Live logging display in admin interfaces
- **Enhanced Metrics**: Real-time statistics updates including dry-run counters
- **Database Schema Migration**: Automatic column addition and schema verification tools

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL (optional, app works without database)
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-service-chatbot.git
cd ai-service-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure secrets**
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-openai-api-key"
VECTOR_STORE_ID = "your-vector-store-id"
ADMIN_PASSWORD = "your-admin-password"
MODEL = "gpt-4o-mini"

# Optional: PostgreSQL configuration
[postgres]
host = "localhost"
port = 5432
user = "your-username"
password = "your-password"
database = "chatbot_db"
```

4. **Run the application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
ai-service-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # License file
├── .streamlit/
│   ├── secrets.toml      # Configuration secrets
│   └── system_prompt.txt # System prompt template
├── assets/
│   └── viadrina-logo.png # University logo
├── css/
│   └── styles.css        # Custom styling
├── utils/
│   ├── __init__.py      # Package initialization
│   └── utils.py         # Database and utility functions
├── pages/
│   ├── admin.py         # Admin dashboard with enhanced settings
│   ├── scrape.py        # Web scraping with LLM analysis & individual controls
│   ├── vectorize.py     # Document processing with terminal-style logging
│   └── view_logs.py     # Analytics dashboard with session tracking
├── tests/                # Test suite (organized for better structure)
│   ├── __init__.py      # Test package initialization
│   ├── README.md        # Test documentation
│   ├── test_filters.py  # Content filtering tests
│   ├── test_session_id.py           # Session tracking verification
│   ├── test_streamlit_sim.py        # Streamlit behavior simulation
│   ├── test_advanced_parameters.py  # API parameter testing
│   ├── test_implementation_status.py # Filter status validation
│   └── test_*.py        # Additional test files
├── exported_markdown/    # Scraped content storage
├── check_db_schema.py   # Database diagnostic and repair tool
├── ux_improvements.md   # Development progress documentation
└── __pycache__/         # Python cache files
```

## � Analytics & Session Tracking

### Conversation Analytics
With the new session tracking system, you can perform advanced analytics:

```sql
-- Count conversations per session
SELECT session_id, COUNT(*) as interaction_count,
       MIN(timestamp) as conversation_start,
       MAX(timestamp) as conversation_end
FROM log_table 
WHERE session_id IS NOT NULL 
GROUP BY session_id
ORDER BY conversation_start DESC;

-- Get full conversation flow
SELECT timestamp, user_input, assistant_response, confidence
FROM log_table 
WHERE session_id = 'your-session-id' 
ORDER BY timestamp;

-- Average conversation length and engagement metrics
SELECT 
    AVG(interaction_count) as avg_conversation_length,
    MAX(interaction_count) as longest_conversation,
    COUNT(DISTINCT session_id) as total_sessions
FROM (
    SELECT session_id, COUNT(*) as interaction_count 
    FROM log_table 
    WHERE session_id IS NOT NULL 
    GROUP BY session_id
) conversation_stats;

-- Response quality by request type
SELECT request_classification,
       AVG(confidence) as avg_confidence,
       COUNT(*) as request_count
FROM log_table 
WHERE confidence > 0
GROUP BY request_classification
ORDER BY avg_confidence DESC;
```

### Key Metrics Available
- **Session Duration**: Time span of individual conversations
- **Conversation Depth**: Number of interactions per session
- **Response Quality**: Confidence scores and error rates by session
- **Query Patterns**: Request classification and topic analysis
- **User Engagement**: Return sessions and interaction patterns

## �🔧 Configuration

### Required Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `VECTOR_STORE_ID`: OpenAI vector store ID for file search
- `ADMIN_PASSWORD`: Password for admin features

### Optional Configuration
- `MODEL`: OpenAI model to use (default: gpt-5-mini)
- PostgreSQL connection details for full functionality

### Database Setup (Optional)

The application works without a database but provides enhanced features with PostgreSQL:

**Local Development:**
```bash
# Install PostgreSQL
createdb chatbot_db
```

**Cloud Deployment:**
- [Neon.tech](https://neon.tech) - 500MB free
- [Supabase](https://supabase.com) - 500MB free
- [ElephantSQL](https://elephantsql.com) - 20MB free

## 🌐 Deployment

### Streamlit Cloud

1. **Push to GitHub**
2. **Connect to Streamlit Cloud**
3. **Add secrets in app settings:**
   - Copy your `.streamlit/secrets.toml` content
   - Configure external PostgreSQL if needed
4. **Deploy**

### Database Migration

To migrate your local knowledge base to cloud:

1. **Backup local database:**
```bash
pg_dump -h localhost -U username chatbot_db > backup.sql
```

2. **Restore to cloud database:**
```bash
psql "postgresql://user:pass@host/db" < backup.sql
```

## 📊 Features Overview

### 🤖 Enhanced Chat Interface
- Clean, responsive design with enhanced visual feedback and progress indicators
- Real-time message streaming with contextual status indicators and visual spinner
- Human-readable progress messages ("Searching the web…", "Searching my knowledge base…", "Generating answer…")
- Source citation with hover details and expandable source lists
- Conversation history with session-based tracking and context maintenance
- Multi-language support with automatic language detection
- Debug mode with detailed response object inspection

### 🔍 Advanced Knowledge Base Management
- **Enhanced Web Scraping** (`/scrape`): LLM-powered content analysis with real-time output
- **Smart Document Processing** (`/vectorize`): Terminal-style logging with live progress updates
- **Individual URL Controls**: Granular save/delete buttons for precise content management
- **Functional Vector Sync**: Working synchronization with 5-minute caching and progress tracking
- **Content Analysis**: LLM-generated summaries and metadata extraction
- **Duplicate Detection**: Content hash-based change detection and update management

### 📈 Comprehensive Analytics & Monitoring
- **Session-Based Analytics**: Conversation tracking with UUID-based session identification
- **Response Evaluation**: Multi-dimensional quality assessment with confidence scoring
- **Request Classification**: Automatic categorization of user queries
- **Real-time Metrics**: Live statistics updates including dry-run counters
- **Error Tracking**: Detailed error classification and logging
- **Performance Monitoring**: Response time and system health tracking
- **Interactive Logs Dashboard** (`/view_logs`): Searchable, filterable interaction history

### 🛠 Enhanced Admin Features
- **System Configuration** (`/admin`): Comprehensive settings management
- **Prompt Versioning**: Historical prompt management with rollback capabilities
- **Authentication System**: Secure admin access with session management
- **Debug Tools**: Advanced diagnostic features and schema verification
- **Database Management**: Automatic schema migration and repair tools
- **Terminal Output**: Live command execution with real-time feedback

## 🔗 API Integration

### OpenAI Integration
- **Responses API**: Primary chat completion
- **File Search**: RAG implementation with vector stores
- **Streaming**: Real-time response generation
- **Evaluation**: Automated response quality assessment

### Database Schema
```sql
-- Main interaction logging with session tracking
CREATE TABLE log_table (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id VARCHAR(36),                    -- UUID for conversation grouping
    user_input TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    error_code VARCHAR(10),
    citation_count INTEGER DEFAULT 0,
    citations JSONB,                          -- Structured citation metadata
    confidence DECIMAL(3,2) DEFAULT 0.0,     -- Response quality score
    request_classification VARCHAR(50),       -- Query type classification
    evaluation_notes TEXT                     -- Detailed evaluation notes
);

-- Knowledge base documents with enhanced metadata
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    summary TEXT,                            -- LLM-generated content summary
    vector_file_id TEXT,                     -- OpenAI vector store file ID
    content_hash VARCHAR(64),                -- Content change detection
    last_updated TIMESTAMP DEFAULT NOW(),
    content_type VARCHAR(50),                -- Document type classification
    word_count INTEGER,                      -- Content length metrics
    processing_status VARCHAR(20) DEFAULT 'pending'
);

-- Prompt management and versioning
CREATE TABLE prompt_versions (
    id SERIAL PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    version_note TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);
```

## 🧪 Development & Testing

### Running Tests
```bash
# Test session ID functionality
python3 tests/test_session_id.py

# Test Streamlit session simulation
python3 tests/test_streamlit_sim.py

# Test filter implementations
python3 tests/test_filters.py

# Validate filter implementation status
python3 tests/test_implementation_status.py

# Verify database schema
python3 check_db_schema.py

# Install development dependencies
pip install -e .

# Run all tests (if using pytest)
pytest tests/
```

### Development Features
- **Session Tracking**: UUID-based conversation analytics
- **Database Diagnostics**: Automatic schema verification and repair
- **Terminal Logging**: Real-time output capture in admin interfaces
- **Debug Mode**: Comprehensive debugging tools and response inspection
- **Schema Migration**: Automatic database updates and column additions

### Performance Optimizations
- **Vector Store Caching**: 5-minute cache for improved response times
- **Streaming Responses**: Real-time message generation with progress indicators
- **Efficient Database Queries**: Optimized logging and retrieval operations
- **Content Deduplication**: Hash-based change detection for updates only

### Code Quality
- **Modular Architecture**: Clean separation of concerns across components
- **Error Handling**: Comprehensive exception management and graceful degradation
- **Documentation**: Inline comments and comprehensive README
- **Configuration Management**: Centralized secrets and environment handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [OpenAI](https://openai.com/) for the powerful AI models
- [Viadrina University](https://www.europa-uni.de/) for the use case and inspiration
- PostgreSQL community for the robust database system

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the [Streamlit documentation](https://docs.streamlit.io/)
- Review [OpenAI API documentation](https://platform.openai.com/docs/)

---

**Built with ❤️ for the Viadrina University Library**