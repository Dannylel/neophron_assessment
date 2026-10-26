# app/request_log.py   ← NEW FILE. generate.py, run_eval.py, web.py are untouched.
import json, time, uuid
from pathlib import Path
from app.generate import answer_question

LOG_PATH = Path("logs/requests.jsonl")

def answer_and_log(query, user_groups, user=None):
    t0 = time.perf_counter()
    result = answer_question(query, user_groups)          
    record = {
        "request_id": str(uuid.uuid4()),
        "ts": time.time(),
        "latency_s": round(time.perf_counter() - t0, 3),
        "user": user, "groups": user_groups, "query": query,
        "retrieved": [(r["chunk_id"], r["score"]) for r in result["retrieved"]],  # IDs only, no text
        "used_tool": result["used_tool"], "tool_calls": result["tool_calls"],
        "grounded": result["grounded"],
        "model_answer": result["model_answer"], "answer": result["answer"],
    }
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:       # "a" = append one line, never rewrite
        f.write(json.dumps(record, default=str) + "\n")
    return result                                         # same output as answer_question