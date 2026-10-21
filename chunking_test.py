import json
from pathlib import Path

folder = Path("documents")
tags = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


for path in sorted(folder.glob("*.txt")):       # Path objects for each .txt file
    # Normalizing line endings handles both Windows (\r\n) and Unix (\n)
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    
    # Split paragraphs and filter out empty ones
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunk_size = 20
    chunk_id = 0 

# Handle completely empty files gracefully
    if not paragraphs:
        print(f"Skipping {path.name}: File is empty.")
        continue

    # Safe lookup for manifest tags to prevent mid-script crashes
    file_acl = tags.get(path.name, {}).get("acl", "UNKNOWN_ACL")

    for para in paragraphs:
        for start in range(0, len(para), chunk_size):
            chunk = para[start:start + chunk_size]
            
            # Print output for EVERY chunk
            print(f"File: {path.name} | Chunk ID: {chunk_id} | ACL: {file_acl} | Text: '{chunk}'")
            
            chunk_id += 1


# for path in sorted(folder.glob("*.txt")):        
#     text = path.read_text(encoding="utf-8")
#     paragraphs = [p.strip() for p in text.split("\\,") if p.strip()]
#     chunk_size=30
#     chunk_id=0

#     for i, para in enumerate(paragraphs):        # enumerate gives (0, para), (1, para), ...
#         for start in range(0, len(para), chunk_size):
#             chunk = para[start:start + chunk_size]
#         # if paragraphs:
#         #     final_byte = paragraphs[-1:]
#             # print(f"Raw byte: {final_byte}")
#         # else:
#         #     print("The file is empty.")
#         print(path.name, chunk_id,tags[path.name]["acl"], para[:30])
#         chunk_id+=1





    