"""
Task 2 -- Deterministic calculation, exposed to the model as a real tool.

NOI must be computed here, in code -- never by the language model. But
unlike a hand-rolled keyword router, the MODEL decides when to call this,
via its own native function-calling interface (OpenAI/Anthropic-style
tool schemas, or Ollama's tool support for models like qwen2.5 / llama3.1).
"""
import csv
import re 
from typing import List, Dict   

def property_directory(csv_path: str = "data/properties.csv") -> str:
    with open(csv_path, newline="") as f:
        names = {int(r["property_id"]): r["name"] for r in csv.DictReader(f)}  # dict = one entry per id
    return ", ".join(f"{pid} = {name}" for pid, name in sorted(names.items()))    

def _normalize_period(period: str) -> str:
    """'Q3 2026', 'q3-2026', 'Q3/2026', '2026 Q3' -> 'Q3-2026'."""
    text = str(period).strip().upper()
    m = re.search(r"Q\s*([1-4])\D*(\d{4})", text)
    if m:
        return f"Q{m.group(1)}-{m.group(2)}"
    m = re.search(r"(\d{4})\D*Q\s*([1-4])", text)
    if m:
        return f"Q{m.group(2)}-{m.group(1)}"
    return text


def calculate_noi(property_id: int, period: str, csv_path: str = "data/properties.csv") -> Dict:
    try:
        pid = int(property_id)
    except (TypeError, ValueError):
        raise ValueError(f"property_id must be an integer, got {property_id!r}")
    wanted_period = _normalize_period(period)
    with open (csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row['property_id'])==pid and _normalize_period(row['period'])==wanted_period:
                result=float(row['revenue'])-float(row['operating_expenses'])
                # return result 
                return {'result': result, 'revenue':float(row['revenue']),
                        'expenses':float(row['operating_expenses']), 'source' : f'{csv_path}#property_id = {pid}, period = {wanted_period}'}
    raise ValueError(f"No data for property {property_id} in {wanted_period}")
    raise NotImplementedError("Read properties.csv and compute NOI deterministically")
CALCULATE_NOI_TOOL_SCHEMA = {
    "name": "calculate_noi",
    "description": (
        "Compute Net Operating Income (revenue - operating expenses) for a "
        "property and period. Always use this tool for NOI questions -- "
        "never estimate or compute NOI yourself."
        " Use this tool ONLY when the user asks for NOI."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "property_id": {"type": "integer", "description": "The property's numeric id"},
            "period": {"type": "string", "description": "e.g. 'Q3-2026'"},
        },
        "required": ["property_id", "period"],
    },
}

# if __name__ == "__main__":
    
#     print(calculate_noi(1, '-2026-q2')) #testing regex python -m eval.run_eval