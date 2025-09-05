# Viadrina Library Assistant 🤖📚

An intelligent AI-powered chatbot built with Streamlit and OpenAI that helps users with library-related questions. The assistant provides contextual answers with source citations using RAG (Retrieval-Augmented Generation) and includes comprehensive logging and evaluation capabilities.

![Viadrina Library Assistant](assets/viadrina-logo.png)

## 🌐 Live Demo

**Try the live application:** [viadrina.streamlit.app](https://viadrina.streamlit.app)

Experience the Viadrina Library Assistant in action! The live demo includes the full chat interface with knowledge base search, web search capabilities, and real-time response streaming.

## ✨ Features

### 🎯 Core Functionality
- **Intelligent Q&A**: Natural language processing for library-related queries
- **Source Citations**: Automatic citation generation with hover tooltips showing summaries
- **Multi-language Support**: Ask questions in any language
- **Real-time Streaming**: Live response generation with human-readable status indicators
- **Enhanced UX**: Visual spinner with contextual status messages ("Searching the web…", "Searching my knowledge base…", "Generating answer…")
- **Conversation Context**: Maintains conversation history for better context understanding

### 📊 Advanced Capabilities
- **Knowledge Base Management**: Web scraping and document ingestion pipeline
- **Response Evaluation**: Automatic classification and quality assessment of responses
- **Admin Dashboard**: Authentication-protected admin features for system management
- **Comprehensive Logging**: Database logging with evaluation metrics and analytics
- **Citation Management**: Smart citation extraction with database-backed metadata

### 🔧 Technical Features
- **RAG Implementation**: Vector search with OpenAI's file search capabilities (displayed as "knowledge base search")
- **Web Search Integration**: External web search with contextual status updates
- **Database Integration**: PostgreSQL with automatic schema migration
- **Enhanced Streaming**: Human-readable progress indicators with visual spinner
- **Error Handling**: Graceful degradation when services are unavailable
- **Cloud-Ready**: Deployment support for Streamlit Cloud with external databases

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
│   ├── admin.py         # Admin dashboard
│   ├── scrape.py        # Web scraping interface
│   ├── vectorize.py     # Document processing
│   └── view_logs.py     # Analytics dashboard
├── exported_markdown/    # Scraped content storage
└── logs/
    └── interaction_log.jsonl # Local logging fallback
```

## 🔧 Configuration

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

### 🤖 Chat Interface
- Clean, responsive design with enhanced visual feedback
- Real-time message streaming with contextual status indicators
- Visual spinner with human-readable progress messages
- Source citation with hover details
- Conversation history

### 🔍 Knowledge Base Management
- Web scraping interface (`/scrape`)
- Document vectorization (`/vectorize`)
- Automatic content processing
- Metadata extraction and storage

### 📈 Analytics & Monitoring
- Response evaluation and classification
- Confidence scoring
- Error tracking
- Interaction logging (`/view_logs`)

### 🛠 Admin Features
- System configuration (`/admin`)
- Prompt management
- Authentication-protected features
- Debug tools and monitoring

## 🔗 API Integration

### OpenAI Integration
- **Responses API**: Primary chat completion
- **File Search**: RAG implementation with vector stores
- **Streaming**: Real-time response generation
- **Evaluation**: Automated response quality assessment

### Database Schema
```sql
-- Main interaction logging
CREATE TABLE log_table (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    user_input TEXT,
    assistant_response TEXT,
    confidence DECIMAL(3,2),
    error_code VARCHAR(10),
    request_classification VARCHAR(50),
    evaluation_notes TEXT,
    citation_count INTEGER,
    citations JSONB
);

-- Knowledge base documents
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    summary TEXT,
    vector_file_id TEXT,
    content_hash VARCHAR(64),
    last_updated TIMESTAMP
);
```

## 🧪 Development

### Running Tests
```bash
# Install development dependencies
pip install -e .

# Run tests (if available)
pytest
```

### Code Structure
- **app.py**: Main application with streaming chat interface
- **utils/utils.py**: Database operations and utility functions
- **pages/**: Streamlit multipage components
- **Response evaluation**: Automated quality assessment system

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