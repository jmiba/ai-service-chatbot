# Viadrina Library Assistant 🤖📚

An intelligent AI-powered library chatbot built with Streamlit and OpenAI that provides research assistance through RAG (Retrieval-Augmented Generation) with domain-restricted web search and comprehensive source citations.

![Viadrina Library Assistant](assets/viadrina-logo.png)

## ✨ Features

### 🎯 Core Intelligence
- **Hybrid RAG System**: Combines local document search with domain-restricted web search
- **Smart Domain Selection**: Automatically selects from 20 specialized academic domains per query
- **Source Citations**: Real-time citation extraction with proper academic referencing
- **Multi-language Support**: Responds in user's language (German, English, Polish, etc.)
- **Streaming Responses**: Live AI response generation with immediate feedback

### 🔍 Advanced Search Capabilities
- **Academic Databases**: Google Scholar, ArXiv, JSTOR, PubMed, Nature, Science
- **Institutional Sources**: University domains, legal databases, medical journals
- **Query-Adaptive Domains**: Medical queries → PubMed/NIH, Legal → EUR-Lex/court records
- **Local Resource Integration**: Direct links to ViaCat for locally available materials

### Enterprise Features  
- **Admin Dashboard**: Authentication-protected system management
- **Response Evaluation**: Automated quality assessment and RAG constraint validation
- **Comprehensive Logging**: PostgreSQL-based analytics with evaluation metrics (required)
- **Prompt Management**: Database-stored system prompts with versioning (required)
- **Knowledge Base Tools**: Web scraping and document vectorization interfaces

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- **PostgreSQL database** (required for all core functionality)
- OpenAI API key with Responses API access
- Vector Store ID (OpenAI file search)

### Installation

1. **Clone and setup**
```bash
git clone https://github.com/yourusername/ai-service-chatbot.git
cd ai-service-chatbot
pip install -r requirements.txt
```

2. **Configure secrets** (`.streamlit/secrets.toml`):
```toml
OPENAI_API_KEY = "your-openai-api-key"
VECTOR_STORE_ID = "your-vector-store-id" 
ADMIN_PASSWORD = "your-admin-password"
MODEL = "gpt-4o-mini"

# Required: PostgreSQL database configuration
[postgres]
host = "localhost"
port = 5432
user = "your-username" 
password = "your-password"
database = "chatbot_db"
```

3. **Run**
```bash
streamlit run app.py
```

## 🧠 RAG Architecture

### Smart Domain Selection
The system automatically selects from **20 specialized domains** per query:

**Core Academic Domains (always included):**
- europa-uni.de, scholar.google.com, arxiv.org
- jstor.org, springer.com, cambridge.org, oxford.org
- sciencedirect.com, wiley.com, taylor-francis.com

**Query-Adaptive Specialization:**
- **Medical**: PubMed, NIH, WHO, Cochrane, NEJM, BMJ, Lancet, JAMA
- **Legal**: EUR-Lex, German legal databases, Yale/Harvard Law, Westlaw
- **Technology**: ACM, IEEE, Stack Overflow, GitHub
- **General Academic**: Nature, Science, PNAS, MIT, Stanford, Harvard

### RAG Workflow
1. **File Search First**: Query local document vector store
2. **Web Search Supplement**: Automatic domain-restricted web search
3. **Citation Extraction**: Process both file_citation and url_citation annotations
4. **Response Synthesis**: Combine sources with proper attribution
5. **Quality Evaluation**: Automated assessment of RAG constraint compliance

## 🔧 Key Configurations

### Domain Specialization
```python
# Automatically detected query types
Medical: "medical", "health", "medicine", "clinical"
Legal: "law", "legal", "court", "legislation" 
Technology: "programming", "software", "AI"
```

### Language Support
- **Universal**: Responds in user's question language
- **Supported**: German, English, Polish, and others
- **Consistent**: No language mixing in responses

### RAG Constraints
- **Source-Only Responses**: No training data or general knowledge
- **Traceable Facts**: Every claim must link to specific sources
- **Transparent Limitations**: Clear statements when information unavailable

## 📁 Project Structure

```
ai-service-chatbot/
├── app.py                    # Main application with RAG pipeline
├── requirements.txt          # Dependencies
├── .streamlit/
│   ├── secrets.toml         # Configuration
│   └── system_prompt.txt    # Fallback system prompt
├── utils/
│   └── utils.py            # Database & utility functions
├── pages/
│   ├── admin.py            # System administration
│   ├── scrape.py           # Web scraping tools
│   ├── vectorize.py        # Document processing
│   └── view_logs.py        # Analytics dashboard
└── assets/css/             # UI components
```

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud  
3. Add secrets from `.streamlit/secrets.toml`
4. **Setup external PostgreSQL** (required - see options below)

### Database Setup (Required)

**Local Development:**
```bash
# Install and setup PostgreSQL
createdb chatbot_db
```

**Cloud Deployment Options:**
- [Neon.tech](https://neon.tech) - 500MB free tier
- [Supabase](https://supabase.com) - 500MB free tier  
- [ElephantSQL](https://elephantsql.com) - 20MB free tier
- [Railway](https://railway.app) - PostgreSQL hosting

### Database Migration
```bash
# Backup local knowledge base
pg_dump chatbot_db > backup.sql

# Restore to cloud
psql "postgresql://user:pass@host/db" < backup.sql
```

## 📊 Analytics & Monitoring

### Response Evaluation
- **Classification**: library_hours, book_search, research_help, etc.
- **Quality Metrics**: Confidence scoring (0.0-1.0)
- **Error Detection**: RAG constraint violations, unsourced content
- **Performance Tracking**: Citation counts, response completeness

### Admin Features (`/admin`)
- **Prompt Management**: Database-stored system prompts with versioning
- **System Configuration**: Debug modes, search parameters
- **User Analytics**: Usage patterns, error analysis
- **Knowledge Base Status**: Document counts, vector store health

## 🔍 Research Integration

### Academic Workflow
1. **Query Processing**: Natural language understanding
2. **Source Selection**: Academic databases + institutional sources  
3. **Information Synthesis**: Multi-source knowledge integration
4. **Citation Generation**: Proper academic referencing
5. **Local Resource Linking**: ViaCat integration for material access

### Supported Research Types
- **Literature Reviews**: Multi-database academic search
- **Current Research**: Recent publications and developments  
- **Institutional Resources**: University-specific information
- **Legal Research**: Court records, legislation, legal databases
- **Medical Research**: Peer-reviewed journals, clinical guidelines

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Test with different query types and languages
4. Ensure RAG constraints maintained
5. Submit Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🏛️ Acknowledgments

- **Viadrina University** for the academic use case
- **OpenAI** for Responses API and vector search capabilities
- **Academic Publishers** for accessible research databases
- **Streamlit** for the excellent web framework

---

**RAG-Powered Research Assistant • Built for Academic Excellence**