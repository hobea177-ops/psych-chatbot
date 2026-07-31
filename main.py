import os
from contextlib import asynccontextmanager
from typing import List, Optional

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import models
from database import Base, engine, get_db

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ضع ANTHROPIC_API_KEY في ملف .env")

client = Anthropic(api_key=ANTHROPIC_API_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Psych Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRUSTED_DOMAINS = [
    "apa.org",
    "who.int",
    "nimh.nih.gov",
    "ncbi.nlm.nih.gov",
    "mayoclinic.org",
    "psychiatry.org",
    "verywellmind.com",
    "cdc.gov",
]

SYSTEM_PROMPT = """أنت مساعد متخصص في تبسيط علم النفس والصحة النفسية للجمهور العربي.
قواعدك:
1. اشرح المفاهيم بلغة عربية واضحة ومبسّطة، مع الدقة العلمية.
2. لا تقدّم تشخيصًا طبيًا أو نفسيًا لأي شخص مهما وصف حالته، ووضّح أنك لست بديلاً عن مختص.
3. إذا ظهرت في الرسالة إشارات لأفكار إيذاء النفس أو أزمة نفسية حادة، توقف عن الشرح المعتاد وشجّع المستخدم بلطف على التواصل مع خط مساعدة أو مختص فورًا.
4. استخدم المصادر المرفقة إليك (إن وجدت) لدعم إجاباتك، واذكر اسم المصدر بشكل طبيعي في النص.
5. إن لم تتوفر مصادر بحث لهذا السؤال، أجب من معرفتك العامة ونبّه أن المصادر أدناه عامة وقد لا تغطي كل التفاصيل.
"""


class Source(BaseModel):
    title: str
    url: str

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    sources: List[Source] = []

    class Config:
        from_attributes = True


class SessionOut(BaseModel):
    id: str
    title: str

    class Config:
        from_attributes = True


class SessionDetail(SessionOut):
    messages: List[MessageOut] = []


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    sources: List[Source]
    session_id: str


def search_trusted_sources(query: str, max_results: int = 4) -> List[Source]:
    if not TAVILY_API_KEY:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": f"{query} psychology mental health",
                "include_domains": TRUSTED_DOMAINS,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            Source(title=r.get("title", r["url"]), url=r["url"])
            for r in data.get("results", [])
        ]
    except Exception:
        return []


def get_or_create_session(db: Session, session_id: Optional[str], first_message: str) -> models.ChatSession:
    if session_id:
        existing = db.get(models.ChatSession, session_id)
        if existing:
            return existing
    title = (first_message[:50] + "…") if len(first_message) > 50 else first_message
    new_session = models.ChatSession(title=title)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    session = get_or_create_session(db, req.session_id, req.message)

    past_messages = (
        db.query(models.Message)
        .filter(models.Message.session_id == session.id)
        .order_by(models.Message.created_at)
        .all()
    )

    sources = search_trusted_sources(req.message)

    context_block = ""
    if sources:
        context_block = "مصادر موثوقة ذات صلة (استخدمها إن كانت مفيدة):\n" + "\n".join(
            f"- {s.title}: {s.url}" for s in sources
        )

    claude_messages = [{"role": m.role, "content": m.content} for m in past_messages]
    user_content = req.message
    if context_block:
        user_content += f"\n\n[سياق للمساعدة]\n{context_block}"
    claude_messages.append({"role": "user", "content": user_content})

    try:
        response = client.mess
