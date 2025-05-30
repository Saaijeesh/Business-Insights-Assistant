import os
import re
import json
import logging
from dotenv import load_dotenv
from google.cloud import firestore
from openai import OpenAI
from langsmith import traceable

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("text_to_query")

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Firestore
db = firestore.Client(database="nosqldb")

@traceable(name="get_feedback_data_from_firestore")
def get_feedback_data_from_firestore(collection_name="feedback_feedback"):
    logger.info(f"Fetching feedback data from Firestore collection: {collection_name}")
    docs = db.collection(collection_name).stream()
    return [doc.to_dict() for doc in docs]

@traceable(name="classify_query_type")
def classify_query_type(user_question):
    prompt = """
    You are an assistant that classifies real estate-related feedback questions into two types:

    1. "analysis" → The question asks for:
    - identifying traits or behaviors (e.g., "Which agent was responsive?")
    - opinions or qualities (e.g., "Who had clear communication?", "Which property had the best reviews?")
    - overall impressions

    2. "filter" → The question asks for:
    - specific entries or records (e.g., "What did people say about property 3?", "Show me feedback for agent 1")

    Return only one word: analysis or filter.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_question}],
            temperature=0
        )
        label = response.choices[0].message.content.strip().lower()
        return label if label in {"analysis", "filter"} else "filter"
    except Exception:
        logger.exception("Failed to classify query type.")
        return "filter"

@traceable(name="classify_feedback_focus")
def classify_feedback_focus(user_question):
    prompt = """
    You identify whether the user is asking about feedback on:
    - a property → return 'property'
    - an agent → return 'agent'
    - or both → return 'both'

    Examples:
    - "What did people say about property 5?" → property
    - "Was agent 3 helpful?" → agent
    - "Feedback on property 2 and the agent" → both

    Return one word only: property, agent, or both.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_question}],
            temperature=0
        )
        result = response.choices[0].message.content.strip().lower()
        return result if result in {"property", "agent", "both"} else "both"
    except Exception:
        logger.exception("Failed to classify feedback focus.")
        return "both"
    
@traceable(name="infer_mentions_and_context")
def infer_mentions_and_context(user_question, chat_memory_dict):
    history = ""
    for i in range(1, len(chat_memory_dict) // 2 + 1):
        history += f"User: {chat_memory_dict[f'q{i}']}\nAssistant: {chat_memory_dict[f'a{i}']}\n"

    prompt = f"""
    You are an assistant that extracts all **mentioned property and agent IDs** from the user's current question and chat history.

    Your response must be a JSON object with:
    - "property_id": a list of property numbers (e.g., [5, 10, 12]) or null if none found
    - "agent_id": a list of agent numbers (e.g., [1, 3]) or null if none found

    Do not return strings like "property 1". Just the numbers in a list.

    If nothing is mentioned, return:
    {{ "property_id": null, "agent_id": null }}

    --- Chat History ---
    {history}
    --- Current Question ---
    {user_question}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        logger.debug(f"LLM mention extraction response: {content}")
        return json.loads(content)
    except Exception:
        logger.exception("Failed to infer mentions.")
        return {"property_id": None, "agent_id": None}



@traceable(name="generate_filter_code")
def generate_filter_code(user_question):
    schema = """
    Translate a natural language question into a Python list comprehension.
    Each entry is a dict with keys:
    - property_id (int), agent_id (int), property_feedback (str), agent_feedback (str)

    Return:
    [entry for entry in data if <condition>]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": schema}, {"role": "user", "content": user_question}],
            temperature=0
        )
        generated_code = response.choices[0].message.content.strip()
        logger.info(f"Generated filter code: {generated_code}")
        return generated_code
    except Exception:
        logger.exception("Failed to generate filter code.")
        return None


def extract_trait_summary(user_question, data, feedback_focus="both", max_entries=100):
    prompt = f"""
    You are an assistant that reads feedback entries and answers the question: "{user_question}"

    Each entry has:
    - property_id
    - agent_id
    - property_feedback
    - agent_feedback

    Summarize and identify specific agents or properties that demonstrate the trait(s) asked (like "clear communication").
    Only refer to agent numbers if necessary (e.g., Agent 3), or say "multiple agents" or "some agents" if general.

    Use only the {max_entries} entries below:
    {json.dumps(data[:max_entries], indent=2)}

    Answer:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("Failed to extract traits.")
        return "Could not identify specific traits from feedback."

@traceable(name="generate_feedback_analysis")
def generate_feedback_analysis(user_question, data, chat_memory_dict, feedback_focus="both", max_entries=120):
    data = data[:max_entries]
    if feedback_focus == "property":
        for entry in data:
            entry.pop("agent_feedback", None)
    elif feedback_focus == "agent":
        for entry in data:
            entry.pop("property_feedback", None)

    history = ""
    for i in range(1, len(chat_memory_dict) // 2 + 1):
        history += f"User: {chat_memory_dict[f'q{i}']}\nAssistant: {chat_memory_dict[f'a{i}']}\n"

    prompt = f"""
    Summarize relevant real estate feedback.

    User question:
    {user_question}

    Chat history:
    {history}

    Feedback data:
    {json.dumps(data, indent=2)}

    Answer clearly and concisely.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("Failed to summarize feedback.")
        return "There was an error while summarizing the feedback."
    
@traceable(name="handle_user_feedback_query")
def handle_user_feedback_query(user_question, collection_name="feedback_feedback", property_mention=None, agent_mention=None, chat_memory_dict=None):
    logger.info(f"Handling feedback query: {user_question}")
    data = get_feedback_data_from_firestore(collection_name)
    if not data:
        return "The feedback dataset is empty."

    mentions = infer_mentions_and_context(user_question, chat_memory_dict or {})
    prop_id_raw = property_mention or mentions.get("property_id")
    agent_id_raw = agent_mention or mentions.get("agent_id")

    # Normalize property_id to a list of integers
    prop_ids = []
    if isinstance(prop_id_raw, str):
        prop_ids = [int(x.strip()) for x in re.findall(r'\d+', prop_id_raw)]
    elif isinstance(prop_id_raw, list):
        prop_ids = [int(x) for x in prop_id_raw if isinstance(x, (str, int))]
    elif isinstance(prop_id_raw, int):
        prop_ids = [prop_id_raw]

    # Normalize agent_id to a list of integers
    agent_ids = []
    if isinstance(agent_id_raw, str):
        agent_ids = [int(x.strip()) for x in re.findall(r'\d+', agent_id_raw)]
    elif isinstance(agent_id_raw, list):
        agent_ids = [int(x) for x in agent_id_raw if isinstance(x, (str, int))]
    elif isinstance(agent_id_raw, int):
        agent_ids = [agent_id_raw]

    query_type = classify_query_type(user_question)
    feedback_focus = classify_feedback_focus(user_question)

    logger.info(f"Query Type: {query_type}, Feedback Focus: {feedback_focus}")
    logger.info(f"Property IDs: {prop_ids}, Agent IDs: {agent_ids}")

    try:
        if query_type == "analysis" and not (prop_ids or agent_ids):
            return extract_trait_summary(user_question, data, feedback_focus)

        filtered_data = []
        if prop_ids and agent_ids:
            filtered_data = [
                e for e in data
                if e.get("property_id") in prop_ids and e.get("agent_id") in agent_ids
            ]
        elif prop_ids:
            filtered_data = [e for e in data if e.get("property_id") in prop_ids]
        elif agent_ids:
            filtered_data = [e for e in data if e.get("agent_id") in agent_ids]
        else:
            code = generate_filter_code(user_question)
            if code:
                logger.info("Using custom filter generated by LLM:")
                logger.info(f"Filter Code:\n{code}")
                try:
                    filtered_data = eval(code, {"data": data})
                except Exception as eval_err:
                    logger.exception("Error evaluating generated filter code.")
                    return f"Generated filter code failed: {str(eval_err)}"

        if filtered_data:
            return generate_feedback_analysis(user_question, filtered_data, chat_memory_dict, feedback_focus)
        return "Couldn't find relevant feedback in the dataset."

    except Exception as e:
        logger.exception("Error during query execution.")
        return f"Error while processing the query: {str(e)}"

