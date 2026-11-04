from pathlib import Path
from typing import Optional, Literal
from datetime import datetime, timezone
import json, logging, re, time, uuid

import chromadb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
ORDERS_PATH = BASE_DIR / "orders.csv"
if not ORDERS_PATH.exists():
    ORDERS_PATH = BASE_DIR / "orders_with_escalation_scores.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"
MEMORY_PATH = BASE_DIR / "conversation_memory.json"
LOG_PATH = BASE_DIR / "api_requests.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "nykaa_sentence_chunks"
TOP_K = 3
GROUNDING_THRESHOLD = 0.3372
DELAY_WEIGHT, RECENCY_WEIGHT, MAX_AGE_DAYS = 0.70, 0.30, 30

if not ORDERS_PATH.exists():
    raise FileNotFoundError(f"Orders CSV not found in {BASE_DIR}")
if not CHROMA_DIR.exists():
    raise FileNotFoundError(f"ChromaDB folder not found in {BASE_DIR}")

orders_df = pd.read_csv(ORDERS_PATH)
required = {"record_id", "status", "order_value_inr", "days_since_created", "delayed_shipment"}
missing = required - set(orders_df.columns)
if missing:
    raise ValueError(f"Missing order columns: {sorted(missing)}")

orders_df["record_id"] = orders_df["record_id"].astype(str).str.strip()
orders_df["status"] = orders_df["status"].astype(str).str.strip()
orders_df["order_value_inr"] = pd.to_numeric(orders_df["order_value_inr"], errors="raise")
orders_df["days_since_created"] = pd.to_numeric(orders_df["days_since_created"], errors="raise")
orders_df["delayed_shipment"] = orders_df["delayed_shipment"].astype(str).str.lower().map({"true": True, "false": False})
if orders_df["delayed_shipment"].isna().any():
    raise ValueError("delayed_shipment must contain only True/False.")

orders_df["escalation_score"] = (
    DELAY_WEIGHT * orders_df["delayed_shipment"].astype(float)
    + RECENCY_WEIGHT * orders_df["days_since_created"] / MAX_AGE_DAYS
).clip(0, 1)
ESCALATION_THRESHOLD = float(orders_df["escalation_score"].quantile(0.80))

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
policy_collection = chroma_client.get_collection(COLLECTION_NAME)
embedding_model = SentenceTransformer(MODEL_NAME)

app = FastAPI(
    title="Nykaa Domain Support Agent API",
    description="Simulated e-commerce support agent for policy questions and order lookups.",
    version="1.0.0"
)

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str = Field(default="api_demo", min_length=1, max_length=100)

class OrderDetails(BaseModel):
    record_id: str
    status: str
    order_value_inr: float

class EscalationDetails(BaseModel):
    score: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1)
    requires_escalation: bool

class AskResponse(BaseModel):
    conversation_id: str
    query: str
    intent: Literal["policy", "order"]
    answer: str
    sources: list[str]
    order_details: Optional[OrderDetails] = None
    escalation: Optional[EscalationDetails] = None
    blocked: bool = False
    trace_id: str

class AddDocumentRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=120)
    source: str = Field(..., min_length=1, max_length=120)
    text: str = Field(..., min_length=20, max_length=12000)

class AddDocumentResponse(BaseModel):
    status: str
    document_id: str
    chunks_added: int
    trace_id: str

PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
CARD_PATTERN = re.compile(
    r"(?i)(\b(?:card|credit\s*card|debit\s*card|payment\s*card)"
    r"\s*(?:last\s*4|last\s*four|ending\s*in|ending|last4)"
    r"\s*(?:(?:is|:|#|-)\s*)?)(\d{4})\b"
)
INJECTION_PATTERNS = [
    re.compile(p, re.I | re.S) for p in [
        r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        r"\boverride\s+(the\s+)?(system|developer)\s+prompt\b",
        r"\breveal\s+(the\s+)?(system|hidden)\s+prompt\b",
        r"\bshow\s+me\s+(your\s+)?(system prompt|hidden instructions)\b",
        r"\bact\s+as\s+(an\s+)?unrestricted\s+(assistant|model)\b"
    ]
]

def mask_pii(text: str) -> str:
    text = PHONE_PATTERN.sub("[PHONE_MASKED]", text)
    return CARD_PATTERN.sub(lambda m: m.group(1) + "[CARD_LAST4_MASKED]", text)

def is_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)

def log_request(trace_id: str, safe_query: str, intent: str, status_code: int, started: float):
    # Only the masked query is written to disk.
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "query": safe_query,
        "intent": intent,
        "status_code": status_code,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2)
    }
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

def retrieve_policy(query: str):
    vector = embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
    result = policy_collection.query(
        query_embeddings=[vector],
        n_results=min(TOP_K, policy_collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    contexts = []
    for doc, meta, distance in zip(
        result.get("documents", [[]])[0],
        result.get("metadatas", [[]])[0],
        result.get("distances", [[]])[0]
    ):
        contexts.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "similarity": round(1 - float(distance), 4)
        })
    return contexts

def lookup_order(record_id: Optional[str]):
    if not record_id:
        return {"found": False, "error": "Please provide your order ID."}
    rows = orders_df[orders_df["record_id"].str.upper() == record_id.upper()]
    if rows.empty:
        return {"found": False, "error": f"No order found for {record_id.upper()}."}
    row = rows.iloc[0]
    score = round(float(row["escalation_score"]), 4)
    return {
        "found": True, "record_id": row["record_id"], "status": row["status"],
        "order_value_inr": float(row["order_value_inr"]), "score": score,
        "requires_escalation": score >= ESCALATION_THRESHOLD
    }

def save_turn(conversation_id: str, query: str, answer: str):
    memory = {}
    if MEMORY_PATH.exists():
        try:
            memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            memory = {}
    history = memory.setdefault(conversation_id, [])
    timestamp = datetime.now(timezone.utc).isoformat()
    history.extend([
        {"role": "user", "content": query, "timestamp": timestamp},
        {"role": "assistant", "content": answer, "timestamp": timestamp}
    ])
    MEMORY_PATH.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "nykaa-domain-support-agent",
            "policy_chunks": policy_collection.count()}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    started = time.perf_counter()
    trace_id = str(uuid.uuid4())
    safe_query = mask_pii(request.query)

    if is_injection(safe_query):
        answer = "I can't process this request because it contains an instruction-override attempt."
        log_request(trace_id, safe_query, "blocked", 400, started)
        return AskResponse(
            conversation_id=request.conversation_id, query=safe_query,
            intent="policy", answer=answer, sources=[], blocked=True, trace_id=trace_id
        )

    order_match = re.search(r"\bORD\d{4,}\b", safe_query, re.I)
    order_terms = ["order status", "track my order", "track order",
                   "where is my order", "check my order", "order update"]
    is_order_query = bool(order_match) or any(t in safe_query.lower() for t in order_terms)

    if is_order_query:
        intent = "order"
        order = lookup_order(order_match.group(0).upper() if order_match else None)
        if not order["found"]:
            answer = order["error"]
            response = AskResponse(
                conversation_id=request.conversation_id, query=safe_query,
                intent="order", answer=answer, sources=[], trace_id=trace_id
            )
        else:
            answer = (
                f"Order {order['record_id']} has status {order['status']}. "
                f"Order value: ₹{order['order_value_inr']:,.2f}. "
                f"Escalation score: {order['score']:.2f}. "
                + ("This order is recommended for human review."
                   if order["requires_escalation"]
                   else "This order does not meet the project's escalation threshold.")
            )
            response = AskResponse(
                conversation_id=request.conversation_id, query=safe_query,
                intent="order", answer=answer,
                sources=["orders dataset (synthetic project data)"],
                order_details=OrderDetails(
                    record_id=order["record_id"], status=order["status"],
                    order_value_inr=order["order_value_inr"]
                ),
                escalation=EscalationDetails(
                    score=order["score"], threshold=round(ESCALATION_THRESHOLD, 4),
                    requires_escalation=order["requires_escalation"]
                ),
                trace_id=trace_id
            )
    else:
        intent = "policy"
        contexts = retrieve_policy(safe_query)
        if not contexts or contexts[0]["similarity"] < GROUNDING_THRESHOLD:
            answer = "I don't know based on the available policy documents."
            sources = []
        else:
            answer = "According to the available project policy documents:\n\n" + "\n\n".join(
                f"[{c['source']}] {c['text']}" for c in contexts
            )
            sources = list(dict.fromkeys(c["source"] for c in contexts if c["source"] != "unknown"))
        response = AskResponse(
            conversation_id=request.conversation_id, query=safe_query,
            intent="policy", answer=answer, sources=sources, trace_id=trace_id
        )

    save_turn(request.conversation_id, safe_query, response.answer)
    log_request(trace_id, safe_query, intent, 200, started)
    return response

@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest):
    trace_id = str(uuid.uuid4())
    chunks = [p.strip() for p in re.split(r"\n\s*\n", request.text) if p.strip()]
    if not chunks:
        raise HTTPException(status_code=422, detail="Document has no usable text.")
    ids = [f"{request.document_id}::api::{i}" for i in range(len(chunks))]
    vectors = embedding_model.encode(chunks, normalize_embeddings=True).tolist()
    metadata = [
        {"source": request.source, "document_id": request.document_id,
         "chunk_index": i, "ingestion_method": "api"}
        for i in range(len(chunks))
    ]
    try:
        policy_collection.upsert(ids=ids, documents=chunks,
                                 embeddings=vectors, metadatas=metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}")
    return AddDocumentResponse(
        status="added", document_id=request.document_id,
        chunks_added=len(chunks), trace_id=trace_id
    )
