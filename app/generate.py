"""
Tasks 1 & 2 -- Real generation, grounding, and model-driven tool-calling.

Wire this up to an ACTUAL language model that supports native tool/function
calling (a hosted API, or a local model via Ollama such as qwen2.5 or
llama3.1) -- not a stub, and not a hand-rolled keyword router. Which model
you use isn't graded; whether you can actually work with a real model's
real behavior is.
"""
import re, csv, calendar
import json
from typing import List, Dict
import ollama
from .retrieval import retrieve, Chunk, is_authorized
from .tools import calculate_noi, CALCULATE_NOI_TOOL_SCHEMA, property_directory
CHAT_MODEL="qwen2.5:7b"
# CALCULATE_NOI_TOOL_SCHEMA.function.name=calculate_noi
# CALCULATE_NOI_TOOL_SCHEMA['function']['name']=calculate_noi
# OLLAMA_TOOLS = [{"type": "function", "function": CALCULATE_NOI_TOOL_SCHEMA}]
_NOI_SCHEMA = {**CALCULATE_NOI_TOOL_SCHEMA,
               "description": CALCULATE_NOI_TOOL_SCHEMA["description"]
               + f" Known properties (id = name): {property_directory()}."}
OLLAMA_TOOLS = [{"type": "function", "function": _NOI_SCHEMA}]





NOI_TOOL_ACL = ["finance"]
def call_model(prompt: str, tools: list | None = None) -> Dict:
    

    """Call your chosen real language model.

    Return a normalized dict shaped like:
        {"text": str | None, "tool_call": {"name": str, "arguments": dict} | None}

    - If the model just answers, `text` is set and `tool_call` is None.
    - If the model wants to call a tool (only possible when `tools` is
      passed), `tool_call` is set and `text` may be None.

    You will need to translate this normalized shape to/from whatever your
    provider's actual API expects (OpenAI, Anthropic, and Ollama all differ
    slightly in envelope). That translation is the point of this task.
    """

    # messages=[{"role": "user", "content": prompt}] if isinstance(prompt,  (str, list)) else prompt
    # messages=[{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    if isinstance(prompt, str):
      messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
      messages = prompt
    else:
      raise TypeError(f"prompt must be a str or a list of messages, got {type(prompt).__name__}")
    response = ollama.chat(
    model=CHAT_MODEL, 
    # messages=[{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt, 
    messages=messages,
    tools=tools, 
    options={"temperature": 0}
    )
    msg=response.message
    if msg.tool_calls:
        first=response.message.tool_calls[0]
        tool_call={
            'name': first.function.name,
            'arguments': first.function.arguments
        }
        text=msg.content or None  #to ensure converting empty strings to None
    else:
        tool_call=None  #local variable to avoid editing the library's response object 
        text=msg.content

      # response2=ollama.chat(model=CHAT_MODEL,messages=messages, tools=OLLAMA_TOOLS)

    return {'text':text,'tool_call':tool_call, 'raw':msg}
    raise NotImplementedError("Wire up a real model, including its tool-calling protocol")


_CITATION = re.compile(r"\[[^\]]*\]")              # [ ... ]  -> removed before extracting numbers
_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")     # 412000 | 412,000 | 154,000.00

def _numbers(text: str) -> set:
    text = _CITATION.sub("", text)
    return {float(n.replace(",", "")) for n in _NUM.findall(text)}

def _mentions_number(answer: str, value: float) -> bool:
    """Tool path: does the narrated answer contain the exact value the tool returned?"""
    return value in _numbers(answer)



_CITED = re.compile(r"\[([^\]]+)\]")

def _known_entities(csv_path="data/properties.csv"):
    with open(csv_path, newline="") as f:
        names = {r["name"] for r in csv.DictReader(f)}        # "Oak Hill Plaza", ...
    short = {n.rsplit(" ", 1)[0] for n in names}              # "Oak Hill", "Maple Ridge", ...
    months = set(calendar.month_name[1:])                     # "January" ... "December"
    return names | short | months

ENTITIES = _known_entities()

def _entities(text: str) -> set:
    text = _CITATION.sub("", text)                            # ignore [doc1.txt#0]
    return {e for e in ENTITIES if re.search(rf"\b{re.escape(e)}\b", text)}
#finds which known names appear in the answer. \b means: whole word only
def is_grounded(answer: str, chunks: List[Chunk]) -> bool:
    """Task 1 grounding check: verify the answer's key claims (names,
    numbers, dates) actually appear in the retrieved chunks it was
    supposedly based on. A simple substring/entity check is enough -- it
    needs to exist, not be sophisticated.
    """
    cited = set(_CITED.findall(answer))
    used = [c for c in chunks if c.chunk_id in cited]       #chunks that are both cited AND retrieved 
    nums = _numbers(answer)
    ents=_entities(answer)
    if not nums and not ents:            #numbers and entities must be present
        return True                      #nothing checkable 
    if not used:
        return False                     #no valid citation

    text = " ".join(c.text for c in used).lower()
    
    evidence_nums = set().union(*(_numbers(c.text) for c in used))
    return nums <= evidence_nums and all(e.lower() in text for e in ents)
    
    raise NotImplementedError("Implement a grounding / hallucination check")



default=["Answer using ONLY the context below and cite chunk ids in brackets, e.g. [doc1.txt#0].",
            "If the context does not contain the answer, say you don't know. "
]
noi_allowed=[
       "For any NOI question, always call the calculate_noi tool, even if the context doesn't mention the property.",
       "Never calculate NOI yourself.Only call calculate_noi for NOI questions. Answer revenue or expense questions from the context, with citations.",
   ]
noi_denied=[
       "This user is not authorized to view NOI figures. If asked about NOI, say so, and never calculate it from the context.",
   ]
def build_sys_prompt(noi_auth: bool) -> str: 
   rules= '\n'.join(default+(noi_allowed if noi_auth else noi_denied))
   return rules
def answer_question(query: str, user_groups: List[str]) -> Dict:
    
    """Orchestration for both tasks:

    - Retrieve authorized context for the query via retrieve().
    - Call the model WITH the calculate_noi tool available
      (CALCULATE_NOI_TOOL_SCHEMA) so the model itself decides whether this
      question needs the tool -- do not pre-classify the query yourself
      with keywords/regex and route around the model.
    - If the model requests the tool call, execute calculate_noi() in code,
      send the result back to the model, and get its final narrated answer.
      The model must never be allowed to just state the NOI number itself.
    - If the model answers directly instead, apply is_grounded() to check
      it against the retrieved chunks.

    Return a dict shaped like:
        {"answer": str, "citations": [...], "grounded": bool, "used_tool": bool}
    """
    allowed=is_authorized(NOI_TOOL_ACL, user_groups)
    tools=OLLAMA_TOOLS if allowed else None     #before tool call
    
    chunks = retrieve(query, user_groups)
    context = "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
    # system = (
    #     "Answer using ONLY the context below and cite chunk ids in brackets, e.g. [doc1.txt#0]. "
    #     "If the context does not contain the answer, say you don't know. "
    #     "Never calculate NOI yourself; NOI must come from the calculate_noi tool."
    #     "NOI questions, always call calculate_noi, even if the context doesn't mention the property.\n\n"
    #     f"Context:\n{context}"
    # )
    
    system = build_sys_prompt(noi_auth=allowed)
    messages = [{"role": "system", "content": system}, 
                {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {query}"}]
    # messages = ['system', {'role':'user','content':query}]

   
    first = call_model(messages, tools)
    tool_log=[]
    if first['tool_call']:
        
        name=first['tool_call']['name']
        arguments=first['tool_call']['arguments']
        result = None
        if name != "calculate_noi" or not allowed:         # AFTER the call, BEFORE execution: code re-checks
          content = "Error: not authorized to use this tool"
        else:
            try:
                result = calculate_noi(**arguments)
                content = result 
            except (ValueError, TypeError) as e:
                content = f"Error: {e}"
        tool_log.append({'name':name, 'arguments':arguments,'result':content})
        

        # messages.append(first['raw'])
        messages.append(first['raw'].model_dump(exclude_none=True))     #converts ollama message object into dict 
        messages.append({'role': 'tool', 'content':json.dumps(content),'tool_name':name})
    # response2 = ollama.chat(model=CHAT_MODEL, messages=messages, tools=OLLAMA_TOOLS)
        final= call_model(messages, None)  #intentionally not passing tools for narration call so the model can’t loop for a tool that doesnt exist 
        #if passed, ollama can cache the prefix  but it should be inside a small loop  
        
        model_answer = final['text'] or ""
        answer = model_answer
        if result is None: 
           grounded = False
           answer = f"I couldn't calculate the NOI:{content}"
        # grounded = isinstance(result, float) and _mentions_number(answer, result)
        else:
           #here i declared 3 sources of truth
            trusted = ({result['result'], result['revenue'],result['expenses']}
                 #1 tool's output
                | _numbers(json.dumps(arguments))    #2 tool-call arguments    
                | set().union(*(_numbers(c.text) for c in chunks)))  #3  retrieved chunks
            grounded = (_mentions_number(answer, result["result"]) 
                        and _numbers(answer) <= trusted) 
            if not grounded:
                        #any figure not found in any authorized chunk (i.e: model-computed roi ) is never shown
                        answer = (f"The NOI for property {arguments['property_id']} in "
                                  f"{arguments['period']} is ${result['result']:,.0f} "
                                  f"(revenue ${result['revenue']:,.0f} − expenses ${result['expenses']:,.0f}).")  
           
    else:
        model_answer = first['text'] or ""
        answer = model_answer
        grounded = is_grounded(answer, chunks)
        if not grounded: 
            answer = "I can't verify that answer against the documents available to you."
        
    return {
        'answer': answer, 
        'citations': [c.chunk_id for c in chunks],
        'grounded': grounded,
        'used_tool': bool(tool_log),
        'tool_calls': tool_log, 
        'model_answer': model_answer, 
        'retrieved': [{'chunk_id':c.chunk_id, 'score': round(c.score, 3), 'text':c.text}for c in chunks],
        'messages': messages,
    }



    # raise NotImplementedError("Implement the full answer_question orchestration")
