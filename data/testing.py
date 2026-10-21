from typing import List
import json

def is_authorized(chunk_acl: List[str], user_groups: List[str]) -> bool:
    """A chunk is authorized for a user if 'all' is in its acl, or if any
    of the user's groups intersects the chunk's acl.
    """
    return 'all' in chunk_acl or any(
        item in chunk_acl for item in user_groups)
    # raise NotImplementedError("Implement the authorization check")


# print(is_authorized(["all"], ["marketing"]))  
# print(is_authorized(["admin"], ["admin"]))  
# print(is_authorized(["admin"], ["marketing"]))  
# print(is_authorized(["admin", "engineering"], ["sales", "engineering"]))
# print(is_authorized([], ["admin"]))

# Load test data from JSON
with open("manifest.json", "r") as file:
    test_data = json.load(file)

# Groups belonging to the user being tested
# user_groups = ("propteam", "finance")
user_groups= 'tomato'
print({user_groups},'\n')
# Test every document
for doc_name, doc_data in test_data.items():
    acl = doc_data["acl"]
    result = is_authorized(acl, user_groups)

    print(f"{doc_name}: {result}")


