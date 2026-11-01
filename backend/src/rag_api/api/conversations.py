"""Independent chat sessions and side-by-side model runs."""

import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openinference.instrumentation import using_session
from sqlalchemy import delete, func, select

from rag_api.api.schemas import NewBranchTurn, NewConversation, NewTurn
from rag_api.db import db_session
from rag_api.models import Conversation, Corpus, LLMCache, Message, Run
from rag_api.observability import observed, record_output
from rag_api.runner import model_name, run_branch
from rag_api.settings import get_settings

router = APIRouter()
ACTIVE_TASKS: set[asyncio.Task] = set()

@router.get("/v1/conversations")
def list_conversations() -> list[dict]:
    """Return recent sessions for the chat sidebar."""
    with db_session() as db:
        conversations = db.scalars(select(Conversation).where(Conversation.id.in_(
            select(Message.conversation_id).where(Message.role == "user")
        )).order_by(
            Conversation.created_at.desc()).limit(100)).all()
        if not conversations:
            return []
        ids = [item.id for item in conversations]
        first_questions: dict[str, str] = {}
        for conversation_id, content in db.execute(select(Message.conversation_id, Message.content).where(
                Message.conversation_id.in_(ids), Message.role == "user", Message.turn_no == 1)):
            first_questions.setdefault(conversation_id, content)
        return [{"conversation_id": item.id, "corpus_id": item.corpus_id,
                 "created_at": item.created_at.isoformat(),
                 "title": first_questions.get(item.id, "New chat")[:72]}
                for item in conversations]

@router.post("/v1/conversations")
def create_conversation(request: NewConversation) -> dict:
    corpus_id = request.corpus_id or get_settings().default_corpus
    with db_session() as db:
        corpus = db.get(Corpus, corpus_id)
        if corpus is None or corpus.status != "ready":
            raise HTTPException(409, f"Corpus {corpus_id} is not ready; ingest files first")
        conversation = Conversation(corpus_id=corpus_id)
        db.add(conversation)
        db.flush()
        return {"conversation_id": conversation.id, "corpus_id": corpus_id}


@router.get("/v1/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    with db_session() as db:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        history = db.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(
            Message.turn_no, Message.created_at, Message.id)).all()
        runs = db.scalars(select(Run).where(Run.conversation_id == conversation_id).order_by(Run.turn_no)).all()
        return {"conversation_id": conversation.id, "corpus_id": conversation.corpus_id,
                "branches": {branch: [{"turn_no": item.turn_no, "role": item.role, "content": item.content}
                                      for item in history if item.branch == branch]
                             for branch in ("slm", "mistral")},
                "runs": {branch: [{"turn_no": item.turn_no, "status": item.status,
                                   "answer": item.answer, "citations": item.citations,
                                   "confidence": item.confidence, "metrics": item.metrics,
                                   "error": item.error}
                                  for item in runs if item.branch == branch]
                         for branch in ("slm", "mistral")}}


@router.delete("/v1/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    with db_session() as db:
        conversation = db.scalar(select(Conversation).where(
            Conversation.id == conversation_id).with_for_update())
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        pending = db.scalar(select(Run.id).where(
            Run.conversation_id == conversation_id, Run.status == "running"))
        if pending:
            raise HTTPException(409, "Conversation still has a running turn")
        db.execute(delete(LLMCache).where(LLMCache.source_conversation_id == conversation_id))
        db.execute(delete(Run).where(Run.conversation_id == conversation_id))
        db.execute(delete(Message).where(Message.conversation_id == conversation_id))
        db.delete(conversation)
    return {"deleted": conversation_id}


def prepare_turn(conversation_id: str, question: str) -> tuple[int, dict[str, str], str]:
    with db_session() as db:
        conversation = db.scalar(select(Conversation).where(Conversation.id == conversation_id).with_for_update())
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        pending = db.scalar(select(Run.id).where(Run.conversation_id == conversation_id, Run.status == "running"))
        if pending:
            raise HTTPException(409, "Previous turn is still running")
        turn_no = (db.scalar(select(func.max(Message.turn_no)).where(Message.conversation_id == conversation_id)) or 0) + 1
        run_ids = {branch: str(uuid4()) for branch in ("slm", "mistral")}
        for branch, run_id in run_ids.items():
            db.add(Message(conversation_id=conversation_id, branch=branch, turn_no=turn_no,
                           role="user", content=question))
            db.add(Run(id=run_id, conversation_id=conversation_id, branch=branch, turn_no=turn_no))
        corpus_id = conversation.corpus_id
    return turn_no, run_ids, corpus_id


@router.post("/v1/conversations/{conversation_id}/turns")
async def compare_turn(conversation_id: str, request: NewTurn) -> dict:
    turn_no, run_ids, corpus_id = prepare_turn(conversation_id, request.question)
    with using_session(conversation_id), observed("rag.compare", "CHAIN", request.question) as span:
        span.set_attribute("conversation_id", conversation_id)
        span.set_attribute("corpus_id", corpus_id)
        span.set_attribute("turn.no", turn_no)
        slm, mistral = await asyncio.gather(
            asyncio.to_thread(run_branch, run_ids["slm"], conversation_id, "slm", corpus_id),
            asyncio.to_thread(run_branch, run_ids["mistral"], conversation_id, "mistral", corpus_id),
        )
        record_output(span, f"{model_name('slm')}: {slm.get('answer') or slm.get('error')}\n"
                            f"{model_name('mistral')}: {mistral.get('answer') or mistral.get('error')}")
    return {"conversation_id": conversation_id, "turn_no": turn_no, "corpus_id": corpus_id,
            "results": {"slm": slm, "mistral": mistral}}


@router.post("/v1/conversations/{conversation_id}/turns/stream")
async def compare_turn_stream(conversation_id: str, request: NewTurn) -> StreamingResponse:
    """SSE events for independent tool progress and visible answer tokens."""
    turn_no, run_ids, corpus_id = prepare_turn(conversation_id, request.question)
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def branch_worker(branch: str) -> dict:
        def publish(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        try:
            result = await asyncio.to_thread(run_branch, run_ids[branch], conversation_id, branch,
                                             corpus_id, publish, True)
        except Exception as exc:  # noqa: BLE001 - keep the other branch streaming
            with db_session() as db:
                run = db.get(Run, run_ids[branch])
                run.status, run.error = "error", str(exc)[:500]
            result = {"status": "error", "run_id": run_ids[branch], "error": str(exc)[:500]}
        await queue.put({"type": "result", "branch": branch, "result": result})
        return result

    async def orchestrate() -> None:
        try:
            with using_session(conversation_id), observed("rag.compare", "CHAIN", request.question) as span:
                span.set_attribute("conversation_id", conversation_id)
                span.set_attribute("corpus_id", corpus_id)
                span.set_attribute("turn.no", turn_no)
                slm, mistral = await asyncio.gather(branch_worker("slm"), branch_worker("mistral"))
                record_output(span, f"{model_name('slm')}: {slm.get('answer') or slm.get('error')}\n"
                                    f"{model_name('mistral')}: {mistral.get('answer') or mistral.get('error')}")
        finally:
            await queue.put({"type": "complete"})

    task = asyncio.create_task(orchestrate())
    ACTIVE_TASKS.add(task)
    task.add_done_callback(ACTIVE_TASKS.discard)

    async def events():
        yield "data: " + json.dumps({"type": "started", "conversation_id": conversation_id,
                                     "turn_no": turn_no, "run_ids": run_ids}) + "\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except TimeoutError:
                event = {"type": "heartbeat"}
            yield "data: " + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"
            if event["type"] == "complete":
                break

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/v1/branches/{branch}/turns")
async def single_branch_turn(branch: str, request: NewBranchTurn) -> dict:
    if branch not in {"slm", "mistral"}:
        raise HTTPException(404, "Unknown branch")
    with db_session() as db:
        conversation = db.scalar(select(Conversation).where(
            Conversation.id == request.conversation_id).with_for_update())
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        pending = db.scalar(select(Run.id).where(
            Run.conversation_id == request.conversation_id, Run.branch == branch, Run.status == "running"))
        if pending:
            raise HTTPException(409, "Previous branch turn is still running")
        turn_no = (db.scalar(select(func.max(Message.turn_no)).where(
            Message.conversation_id == request.conversation_id, Message.branch == branch)) or 0) + 1
        run_id = str(uuid4())
        db.add(Message(conversation_id=request.conversation_id, branch=branch, turn_no=turn_no,
                       role="user", content=request.question))
        db.add(Run(id=run_id, conversation_id=request.conversation_id, branch=branch, turn_no=turn_no))
        corpus_id = conversation.corpus_id
    result = await asyncio.to_thread(run_branch, run_id, request.conversation_id, branch, corpus_id)
    return {"conversation_id": request.conversation_id, "turn_no": turn_no,
            "corpus_id": corpus_id, "branch": branch, "result": result}


@router.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict:
    with db_session() as db:
        run = db.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return {"run_id": run.id, "branch": run.branch, "turn_no": run.turn_no, "status": run.status,
                "answer": run.answer, "citations": run.citations, "confidence": run.confidence,
                "metrics": run.metrics, "error": run.error}
