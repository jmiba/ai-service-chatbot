# Viadrina Library Assistant 🤖📚

An AI-powered Streamlit app using OpenAI Responses API. It answers university- and library-related questions with Retrieval-Augmented Generation (RAG), web search, and rich citations. Includes an admin UI for settings, prompt versioning, filters, analytics, and scraping/vectorization workflows.

![Viadrina Library Assistant](assets/viadrina-ub-logo.png)

## 🌐 Live Demo

Try it: https://viadrina.streamlit.app

## ✨ Highlights

- **Multilingual chat assistant** with streaming Responses API output, status updates, and resilient error handling.
- **Retrieval-Augmented Generation** backed by the documents knowledge base (PostgreSQL + OpenAI File Search) with inline citations.
- **One-click document viewer** that opens cited internal markdown in a dedicated Streamlit tab (no more losing context).
- **Web search tooling** with admin-managed allow lists, locale/user-location overrides, and per-request cost estimates.
- **Admin control center** covering prompt versioning, LLM settings, request classifications, web search filters, and knowledge-base maintenance.
- **Content ingestion pipeline** with scraping, manual metadata editing, SHA hashing, and optional vector store sync.
- **Observability out of the box**: detailed interaction logging, usage/cost tracking, request analytics, and job locking.

## 🆕 What’s New (recent changes)

- Web search settings moved to Admin → Filters:
  - Allowed Domains (optional): restrict web search to these domains when set; if empty, web search is unrestricted
  - User Location: type (approximate/precise), country, city, region (optional), timezone (optional)
  - Note: The API does not support exclude-only lists or a locale filter
- Request Classifications are now DB-backed and editable in Admin → Request Classes
- Internal citations now open in a dedicated `/document_viewer` tab so staff can inspect markdown sources without leaving the chat
- Sidebar "New chat" button directly beneath "Chat Assistant"
- Client-managed conversation context retained for predictability (see Context section)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- PostgreSQL (for logging/admin features)

### Install

```bash
pip install -r requirements.txt
```

### Configure secrets
Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-openai-api-key"
VECTOR_STORE_ID = "your-vector-store-id"
ADMIN_PASSWORD = "your-admin-password"
# Optional: default model (used only if DB LLM settings unavailable)
MODEL = "gpt-4o-mini"
# Optional: shown as default in Admin UI and for auditing
ADMIN_EMAIL = "you@your.org"

[postgres]
host = "your-host"   # optional (when omitted, app runs in reduced mode)
port = 5432
user = "your-user"
password = "your-password"
database = "chatbot_db"
```

Run:
```bash
streamlit run app.py
```

## 🔧 Configuration Notes

### Web Search (Admin → Filters)
- Allowed Domains (optional)
  - Provide one or more domains (e.g., `arXiv.org`) to restrict search
  - If the list is empty, the app sends no filters and web search runs unrestricted (when enabled)
- User Location (optional fields)
  - type: `approximate` or `precise`
  - country: ISO 3166-1 alpha-2 (e.g., `DE`, `US`)
  - city: free text (e.g., `Frankfurt (Oder)`, `New York`)
  - region: free text
  - timezone: IANA name (e.g., `Europe/Berlin`, `America/New_York`)

Example tool payload the app generates (simplified):
```json
{
  "type": "web_search",
  "filters": {
    "allowed_domains": ["arXiv.org", "jstor.org", "data.europa.eu", "harvard.edu"]
  },
  "user_location": {
    "type": "approximate",
    "country": "DE",
    "city": "Frankfurt (Oder)",
    "region": "Brandenburg",
    "timezone": "Europe/Berlin"
  }
}
```

### Request Classifications (Admin → Request Classes)
- Categories are stored in PostgreSQL (`request_classifications` table)
- The UI ensures `other` is always present; you can edit the list freely

### LLM Settings (Admin → Language Model)
- Model, parallel tool calls, optional reasoning effort/verbosity (if supported)
- Settings are persisted in PostgreSQL (`llm_settings` table)

## 💬 Context & Conversations

This app currently uses client-managed context for reliability and control:
- On each turn we send the system prompt + the last N turns of history + the new user message
- This keeps responses deterministic and makes trimming explicit
- Responses API returns `conversation: null` because server-side conversations are not used

If you want to switch to OpenAI-managed conversations later:
- Create a conversation once, store `conversation_id` in session state
- Pass `conversation = conversation_id` on each call and only send the new user message
- Widen DB columns if you want to store the OpenAI conversation ID (longer than UUID-36)

## 📚 Features Overview

- Streaming responses with readable statuses ("Searching the web…", etc.)
- RAG with File Search: vector store-backed citations with hover tooltips
- Dedicated `document_viewer` page for browsing cited markdown with summaries, tags, and back-link
- Admin: scraping, vectorization, prompt versioning, filters, logs/analytics
- Session tracking: UUID-based sessions with costs/usage/latency logging
- Robust error handling and graceful degradation without DB

## 🗃️ Database

- Auto-creates tables on first run when `[postgres]` secrets are present and accessible
- Key tables:
  - `log_table`: full interaction logs (session_id, citations, costs, latency, etc.)
  - `prompt_versions`: prompt history
  - `llm_settings`: model/configuration in DB
  - `filter_settings`: web_search enable flag, optional allowed domains, and user location
  - `request_classifications`: editable list of request classes

## 📦 Project Structure (short)

```
ai-service-chatbot/
├── app.py              # Main Streamlit app (chat)
├── pages/              # Admin + tools (scrape, vectorize, logs)
├── utils/              # DB and helper functions
├── css/, assets/       # UI assets
├── tests/              # Test suite
└── .streamlit/         # secrets.toml, prompts.json
```

## 🌐 Deployment (Streamlit Cloud)

- Push to GitHub → Deploy on Streamlit Cloud
- Add your `.streamlit/secrets.toml` in app settings
- For DB-backed features, configure an external PostgreSQL (Neon, Supabase, etc.)

## 🧪 Tests

See `tests/` for examples covering filters, session handling, and UI behavior.

## 📝 License

MIT — see `LICENSE`.

## 🙋 Support

- Open a GitHub issue
- Streamlit docs: https://docs.streamlit.io/
- OpenAI docs: https://platform.openai.com/docs/

---

**Built with ❤️ for the Viadrina University Library**
