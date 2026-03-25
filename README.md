# PDF RAG App

This project now includes:

- a FastAPI backend for agents, uploads, chat, conversations, and settings
- a local Qdrant vector store
- a Next.js + React + Tailwind SaaS admin panel frontend

## Backend

The backend:

- manages chatbot agents
- uploads and indexes PDFs per agent
- stores vectors in local Qdrant collections
- saves conversations and bot settings locally
- answers questions with `gpt-4o-mini`

### Backend setup

1. Create `.env` from `.env.example`.
2. Set `OPENAI_API_KEY`.
3. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

5. Start FastAPI:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Backend URLs:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## Frontend

The frontend lives in `frontend/` and includes the requested component structure:

- `components/Sidebar.tsx`
- `components/Topbar.tsx`
- `components/AgentCard.tsx`
- `components/ChatWidget.tsx`
- `components/MessageBubble.tsx`
- `components/FileUpload.tsx`

App routes include:

- `/dashboard`
- `/agents`
- `/agents/[agentId]/knowledge`
- `/agents/[agentId]/chat`
- `/agents/[agentId]/conversations`
- `/agents/[agentId]/settings`

### Frontend setup

1. In `frontend/`, copy `.env.local.example` to `.env.local` if needed.
2. Install dependencies:

```powershell
cd frontend
"C:\Program Files\nodejs\npm.cmd" install
```

3. Start the frontend:

```powershell
"C:\Program Files\nodejs\npm.cmd" run start -- --hostname 127.0.0.1 --port 3000
```

Frontend URL:

- `http://127.0.0.1:3000`

## Core API

- `GET /dashboard/summary`
- `GET /agents`
- `POST /agents`
- `DELETE /agents/{agent_id}`
- `POST /upload-pdf`
- `GET /agents/{agent_id}/documents`
- `GET /agents/{agent_id}/conversations`
- `GET /agents/{agent_id}/settings`
- `PUT /agents/{agent_id}/settings`
- `POST /chat`

Example chat request:

```json
{
  "question": "What does the document say about scholarships?",
  "agent_id": "your-agent-id"
}
```

Response:

```json
{
  "answer": "...",
  "conversation_id": "..."
}
```
