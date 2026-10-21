import json
from app.generate import answer_question
from app.users import TEST_USERS





GOLD = {"q1": "doc1.txt", "q2": "doc3.txt", "q3": "doc2.txt", "q4": "doc4.txt", "q5": "doc6.txt"}

def reciprocal_rank(chunk_ids, gold_doc):
    for rank, cid in enumerate(chunk_ids, start=1):   # enumerate gives (1, first), (2, second), ...
        if cid.split("#")[0] == gold_doc:             # "doc1.txt#0" -> "doc1.txt"
            return 1 / rank
    return 0.0  

with open("eval/queries.json") as f:
    queries = json.load(f)

results = []
for q in queries:
    r = answer_question(q["query"], TEST_USERS[q["user"]])
    r["reciprocal_rank"] = reciprocal_rank(r["citations"], GOLD[q["id"]])
    results.append({**q, **r})                 # merge: id/user/query + the pipeline output
    print(q["id"], "| tool:", r["used_tool"], r["tool_calls"], "| grounded:", r["grounded"])
    print("   top:", [(x["chunk_id"], x["score"]) for x in r["retrieved"]])
    print("   answer:", r["answer"])
mrr = sum(x["reciprocal_rank"] for x in results) / len(results)
print("MRR:", mrr)
with open("eval/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)  # default=str: anything not JSON-able becomes text



