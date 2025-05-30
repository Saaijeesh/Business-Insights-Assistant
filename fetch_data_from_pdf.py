import os
import re
import json
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from faiss_setup import create_faiss_indexes_from_folder
from openai import OpenAI
from langsmith import traceable

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Init
client = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
faiss_index = create_faiss_indexes_from_folder()


# --- Decide whether to use chat history ---
@traceable(name="should_use_chat_history")
def should_use_chat_history(user_question):
    system_prompt = """
You are an assistant that decides whether a user's question requires previous chat history to be understood.

Instructions:
- If the question contains vague references like "those properties", "the one mentioned earlier", "this property", "it", or "them", return "yes".
- If the question is standalone or general like "what do customers say in general", "list top properties", "what are agent reviews", return "no".
- Ignore phrases like "in general", "overall", "typically" — they do NOT require chat history unless tied to a specific earlier reference.

ONLY return "yes" or "no".
"""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0)
        result = response.choices[0].message.content.strip().lower()
        return result == "yes"
    except Exception as e:
        logger.warning(f"[FAISS] Error checking memory use: {e}")
        return False


# --- Extract property mentions ---
@traceable(name="extract_property_mentions")
def extract_property_mentions(user_question, chat_memory_dict=None):
    system_prompt = """
    You are an assistant that extracts property references from the user's current question and the past conversation history.

    Instructions:
    - Return property identifiers only in the form: "property 1", "property 2", etc.
    - If the user's question is vague or references previously discussed properties, then extract the relevant property IDs from the chat history.
    - If the user's question is standalone and doesn't reference earlier context, extract any property numbers explicitly stated in the question.
    - If nothing is mentioned or implied, return an empty list.

    Respond strictly with a JSON list, like:
    ["property 5", "property 9"]
    or
    []
    """

    messages = [{"role": "system", "content": system_prompt}]
    if chat_memory_dict:
        history = "\n".join([f"User: {chat_memory_dict[f'q{i}']}\nAssistant: {chat_memory_dict[f'a{i}']}" 
                             for i in range(1, len(chat_memory_dict)//2 + 1)])
        logger.info(f"[FAISS] Passing chat history:\n{history}")
        messages.append({"role": "user", "content": f"{history}\n{user_question}"})
    else:
        messages.append({"role": "user", "content": user_question})

    try:
        logger.info(f"[FAISS] Extracting property mentions from: {user_question}")
        response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0)
        raw_content = response.choices[0].message.content.strip()
        matches = re.findall(r'property\s*(?:id)?\s*(\d+)', raw_content.lower())
        cleaned = [f"property {m}" for m in matches]
        logger.info(f"[FAISS] Cleaned property mentions: {cleaned}")
        return cleaned
    except Exception as e:
        logger.warning(f"[FAISS] Failed to extract mentions: {e}")
        return []


# --- Search FAISS PDFs ---
@traceable(name="search_property_pdfs")
def search_property_pdfs(faiss_indexes, user_question, property_mention=None, chat_memory_dict=None, max_results=3):
    logger.info(f"[FAISS] Searching PDFs for: {user_question}")
    docs = []
    doc_sources = []

    use_memory = should_use_chat_history(user_question)
    logger.info(f"[FAISS] Use chat history? {use_memory}")

    memory_to_use = chat_memory_dict if use_memory else None
    mentions = extract_property_mentions(user_question, memory_to_use)

    explicit_mentions = re.findall(r'property\s*(?:id)?\s*\d+', user_question.lower())
    if not mentions and not explicit_mentions:
        mentions = []

    logger.info(f"[FAISS] Final mentions to search: {mentions}")

    valid_keys = []
    for mention in mentions:
        match = re.search(r"property\s*(?:id)?\s*(\d+)", mention)
        if match:
            valid_keys.append(f"Property_ID_{match.group(1)}")

    if valid_keys:
        logger.info(f"[FAISS] Valid keys: {valid_keys}")
        for key in valid_keys:
            if key in faiss_indexes:
                results = faiss_indexes[key].similarity_search(user_question, k=2)
                docs.extend(results)
                doc_sources.extend([(key, d.page_content) for d in results])
            else:
                logger.warning(f"[FAISS] Property key not found: {key}")
    else:
        logger.warning("[FAISS] No valid property IDs found. Running similarity search across all properties.")
        all_ranked = []

        for pid, idx in faiss_indexes.items():
            results = idx.similarity_search(user_question, k=2)
            for res in results:
                all_ranked.append((pid, res))

        # Collect top N properties
        unique_properties = {}
        for pid, doc in all_ranked:
            if pid not in unique_properties and len(unique_properties) < max_results:
                unique_properties[pid] = [doc.page_content]
            elif pid in unique_properties and len(unique_properties[pid]) < 2:
                unique_properties[pid].append(doc.page_content)

        for pid, contents in unique_properties.items():
            for c in contents:
                docs.append(c)
                doc_sources.append((pid, c))

    if not docs:
        return {"error": "No matching documents found."}

    grouped = {}
    for pid, content in doc_sources:
        grouped.setdefault(pid, []).append(content)

    context = ""
    for pid, chunks in grouped.items():
        context += f"\nPROPERTY {pid}:\n" + "\n".join(chunks[:2]) + "\n"

    logger.info("[FAISS] Assembled context for answer.")
    return {"context": context.strip(), "docs": docs}


# --- Extract requested count ---
def extract_requested_count_via_llm(user_question: str, client, default=3) -> int:
    system_prompt = """
You are an assistant that extracts how many properties the user is asking for in their question.

- If they mention an exact number, return that number.
- If no number is mentioned, return the default: 3.

Respond with only the number. No words, no explanation.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r'\d+', content)
        return int(match.group()) if match else default
    except Exception as e:
        print(f"LLM count extraction error: {e}")
        return default


# --- Generate final answer from context ---
@traceable(name="generate_answer_from_context")
def generate_answer_from_context(user_question, context, client):
    requested_count = extract_requested_count_via_llm(user_question, client, default=3)
    
    prompt = f"""
You are a real estate assistant that answers property-related questions using the provided PDF excerpts.

Your goal is to select ONLY the properties that clearly satisfy the user's request. For example, if the user asks for properties with a "community hall", include only those that explicitly mention having one.

Question:
{user_question}

Context:
\"\"\"{context}\"\"\"

Instructions:
- Return exactly {requested_count} properties that match the user's query.
- DO NOT include properties that do not match the requested feature.
- If fewer than {requested_count} properties match, return as many as available.
- Keep your response concise and clearly formatted.

Answer:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("[FAISS] Error generating final answer")
        return "There was an error while generating the answer."
