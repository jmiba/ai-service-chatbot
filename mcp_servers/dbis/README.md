# DBIS MCP Server

This MCP server exposes a few DBIS API endpoints for use inside the chatbot.

## Available tools

- `dbis_list_resource_ids` – `/resourceIds/organization/{organizationId}`
- `dbis_get_resource` – `/resource/{resourceId}/organization/{organizationId}`
- `dbis_list_subjects` – `/subjects`
- `dbis_list_resource_ids_by_subject` – `/resourceIdsBySubject/{subjectId}/organization/{organizationId}`

Responses are returned as JSON payloads so the chatbot can decide how to surface the information.

## Running locally

```bash
pip install -r requirements.txt
python mcp_servers/dbis/server.py
```

The server listens on the default MCP stdio transport. Configure your MCP client with:

```json
{
  "type": "stdio",
  "command": "python",
  "args": ["mcp_servers/dbis/server.py"]
}
```

If you run the MCP server in a different environment, make sure to expose the command path accordingly.

## Notes

- The wrapper uses `httpx` under the hood and applies reasonable timeouts.
- DBIS does not require API keys but does enforce rate limits, so avoid hammering the same endpoint.
