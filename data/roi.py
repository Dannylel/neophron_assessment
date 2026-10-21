import csv
import re 



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


def calculate_noi(property_id: int, period: str, csv_path: str = "properties.csv") -> float:
    try:
        pid = int(property_id)
    except (TypeError, ValueError):
        raise ValueError(f"property_id must be an integer, got {property_id!r}")
    wanted_period = _normalize_period(period)
    with open (csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row['property_id'])==pid and _normalize_period(row['period'])==wanted_period:
                result=float(row['revenue'])-float(row['operating_expenses'])
                return result
    raise ValueError(f"No data for property {property_id} in {wanted_period}")
    
if __name__ == "__main__":
    
    print(calculate_noi(1, '-2026-q2')) #testing regex
    # print(_normalize_period('2026-q2'))