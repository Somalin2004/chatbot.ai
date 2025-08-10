# app.py
import os
import time
import threading
import webbrowser
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime

# Gemini SDK
from google import genai
from google.genai import types

# Load .env
load_dotenv()

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CHATBOT.AI")

# Paths
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
INDEX_FILE = os.path.join(FRONTEND_DIR, "index.html")

# API key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("⚠ GEMINI_API_KEY not found in environment.")

# Init client
client = None
try:
    if API_KEY:
        client = genai.Client(api_key=API_KEY)
        logger.info("✅ CHATBOT.AI client initialized")
    else:
        logger.error("❌ No API key provided, cannot init client.")
except Exception as e:
    logger.error(f"❌ Error initializing CHATBOT.AI client: {e}")

# FastAPI app
app = FastAPI(
    title="Enhanced CHATBOT.AI Chatbot API",
    description="Advanced AI chatbot with comprehensive response capabilities",
    version="2.1.1"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Static files
static_dir = os.path.join(FRONTEND_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"📁 Serving static from {static_dir}")

# Serve index.html
@app.get("/")
async def index():
    if not os.path.isfile(INDEX_FILE):
        raise HTTPException(status_code=500, detail="index.html not found")
    return FileResponse(INDEX_FILE)

# Conversation manager
class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.max_history_turns = 15
        self.max_context_length = 8000

    def add_message(self, session_id: str, role: str, text: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append({
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_history(session_id)

    def _trim_history(self, session_id: str):
        history = self.conversations[session_id]
        if len(history) > self.max_history_turns * 2:
            self.conversations[session_id] = history[-self.max_history_turns * 2:]
        total_chars = sum(len(msg["text"]) for msg in self.conversations[session_id])
        while total_chars > self.max_context_length and len(self.conversations[session_id]) > 2:
            removed = self.conversations[session_id].pop(0)
            total_chars -= len(removed["text"])

    def get_history(self, session_id: str) -> List[Dict]:
        return self.conversations.get(session_id, [])

    def clear_session(self, session_id: str):
        self.conversations.pop(session_id, None)

conv_manager = ConversationManager()

# Prompt builder
def build_enhanced_prompt(message: str, history: List[Dict]) -> str:
    system_instructions = """You are an advanced AI assistant.
Always provide complete, thorough, well-structured answers in Markdown.

Guide:
- Fully answer without skipping details.
- Use headings, bullet points, and numbering.
- Give examples and practical applications.
- Explain both 'why' and 'what'.
- Use code blocks for code, tables for data.
- Anticipate follow-up questions.

Do NOT end early. Always finish your explanation.
"""
    parts = [system_instructions, "\nConversation History:"]
    for turn in history[-10:]:
        role = "Human" if turn["role"] == "user" else "Assistant"
        parts.append(f"\n{role}: {turn['text']}")
    parts.append(f"\nHuman: {message}")
    parts.append("\nAssistant:")
    return "\n".join(parts)

# Auto-continue response generator
def generate_full_response(prompt: str, max_loops: int = 3) -> str:
    full_reply = ""
    loops = 0
    while loops < max_loops:
        loops += 1
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt if loops == 1 else "Please continue your previous answer in full detail.",
            config=types.GenerateContentConfig(
                max_output_tokens=2048,
                temperature=0.3,
                top_p=0.85,
                top_k=40
            )
        )
        text = getattr(response, "text", "").strip()
        if not text:
            break
        full_reply += ("\n" if full_reply else "") + text
        # Stop if this chunk is small or ends cleanly
        if len(text) < 50 or text.endswith((".", "?", "!", "```")):
            break
    return full_reply

# Chat endpoint
@app.post("/api/chat")
async def chat_endpoint(req: Request):
    try:
        body = await req.json()
        message = str(body.get("message", "")).strip()
        session_id = body.get("session_id", "default")
        if not message:
            raise HTTPException(status_code=400, detail='Missing "message"')
        logger.info(f"[{session_id}] User: {message[:100]}{'...' if len(message) > 100 else ''}")

        conv_manager.add_message(session_id, "user", message)
        history = conv_manager.get_history(session_id)
        prompt = build_enhanced_prompt(message, history)

        if not client:
            raise RuntimeError("CHATBOT.AI client not initialized.")

        logger.info(f"[{session_id}] Generating response...")
        reply = generate_full_response(prompt)
        if not reply:
            reply = "⚠ I couldn't generate a full answer. Please try rephrasing."
            logger.warning(f"[{session_id}] Empty reply from CHATBOT.AI.")

        conv_manager.add_message(session_id, "assistant", reply)
        logger.info(f"[{session_id}] Reply length: {len(reply)} chars")

        return JSONResponse({
            "reply": reply,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(conv_manager.get_history(session_id))
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))

# Embedding endpoint
@app.post("/api/embed")
async def embed_endpoint(req: Request):
    try:
        body = await req.json()
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail='Missing "text"')
        if not client:
            raise RuntimeError("CHATBOT.AI client not initialized.")
        logger.info(f"Embedding text length {len(text)}")
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=[text]
        )
        emb = result.embeddings[0].values
        return JSONResponse({
            "embedding": emb,
            "dimension": len(emb),
            "text_length": len(text)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Embedding error")
        raise HTTPException(status_code=500, detail=str(e))

# Clear session endpoint
@app.post("/api/clear-session")
async def clear_session_endpoint(req: Request):
    try:
        body = await req.json()
        session_id = body.get("session_id", "default")
        conv_manager.clear_session(session_id)
        logger.info(f"[{session_id}] Session cleared")
        return JSONResponse({"success": True, "message": f"Session {session_id} cleared."})
    except Exception as e:
        logger.exception("Clear session error")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/api/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_client": "initialized" if client else "not_initialized",
        "active_sessions": len(conv_manager.conversations)
    })

# Run locally
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    url = f"http://127.0.0.1:{port}/"
    logger.info(f"🚀 Starting Chatbot at {url}")

    def open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if os.getenv("AUTO_OPEN_BROWSER", "1") == "1":
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("app:app", host=host, port=port,
                log_level="info",
                reload=os.getenv("DEBUG", "0") == "1")
