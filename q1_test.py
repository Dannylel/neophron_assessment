from app.generate import answer_question

r = answer_question("What was Maple Ridge Apartments' total revenue in Q3 2026?", ["finance"])
print("ANSWER:", r["answer"])