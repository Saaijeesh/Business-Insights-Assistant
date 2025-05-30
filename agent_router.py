import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable
from sqlite_setup import get_sqlite_conn
from text_to_sql import handle_user_question
from text_to_query import handle_user_feedback_query
from fetch_data_from_pdf import search_property_pdfs, generate_answer_from_context, faiss_index

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Define the function schema for routing queries
functions = [
    {
        "name": "route_query",
        "description": "Route the user's question to the correct data source and extract inferred mentions.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "enum": ["faiss", "sql", "firestore"],
                    "description": "Choose: faiss (PDF), sql (CSV), or firestore (JSON feedback)"
                },
                "property_mention": {
                    "type": "string",
                    "description": "Mentioned or inferred property ID (e.g., 'property 5')"
                },
                "agent_mention": {
                    "type": "string",
                    "description": "Mentioned or inferred agent ID (e.g., 'agent 3')"
                }
            },
            "required": ["destination"]
        }
    }
]

def is_general_message(user_message):
    prompt = f"""
    You are a classification assistant.

    Decide if this message is general/small talk (e.g., greetings, how are you, what's up), or a business question about properties, real estate, or agents.

    Message: "{user_message}"

    Reply with one word only: "general" or "business"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower() == "general"
    except Exception as e:
        logger.warning(f"Failed to classify message type: {e}")
        return False

def generate_friendly_reply(user_message):
    prompt = f"""
    You are a friendly and professional assistant that helps business owners manage and analyze the performance of their real estate portfolio.

    The user said:
    "{user_message}"

    Respond in one of the following ways:

    1. If the message is a greeting or small talk (e.g., "Hi", "How’s it going"), politely acknowledge it and then guide the user to ask a property-related question — such as analyzing top-selling properties, understanding feedback, or checking agent performance.

    2. If the user is asking about what datasets or information you have (e.g., "What kind of datasets do you have", "What do you know", "What can you do"), clearly explain that you have access to:
    - Property sales data (CSV)
    - Customer and agent feedback (JSON)
    - Property descriptions and amenities (PDFs)

    Mention that you can help analyze best-selling properties, agent performance, feedback trends, and overall business performance.

    Keep the tone helpful, business-focused, and brief.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Failed to generate friendly reply: {e}")
        return (
            "Hello! I'm here to help you analyze your property portfolio — "
            "from sales performance to agent effectiveness and customer feedback. "
            "Let me know what you'd like to explore today."
        )


@traceable(name="route_query_with_function_call")
def route_query_with_function_call(user_question, chat_memory_dict=None):
    logger.info("Routing the query...")
    system_prompt = """
    You are a routing agent that determines the best backend (faiss, sql, firestore) to answer the user's question.

    - 'faiss' → Use for property descriptions, amenities, furnishing (PDFs)
    - 'sql' → Use for property prices, sale data, cities, beds/baths (CSV)
    - 'firestore' → Use for agent/property feedback (JSON)

    Use chat history to resolve vague references and infer property/agent mentions.
    """

    messages = [{"role": "system", "content": system_prompt}]
    if chat_memory_dict:
        for i in range(1, len(chat_memory_dict) // 2 + 1):
            messages.append({"role": "user", "content": chat_memory_dict[f"q{i}"]})
            messages.append({"role": "assistant", "content": chat_memory_dict[f"a{i}"]})
    messages.append({"role": "user", "content": user_question})

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=functions,
            function_call={"name": "route_query"},
            temperature=0
        )
        route_info = json.loads(response.choices[0].message.function_call.arguments)
        logger.info(f"Routing Decision: {route_info}")
        return route_info
    except Exception as e:
        logger.exception(f"Failed to route query: {e}")
        return {"destination": "sql", "property_mention": None, "agent_mention": None}

@traceable(name="generate_natural_answer")
def generate_natural_answer(user_question, structured_data):
    logger.info("Generating human-readable answer from structured data...")

    # Truncate to avoid token overflow
    try:
        if isinstance(structured_data, list):
            structured_data = structured_data[:10]
            if structured_data:
                keys_to_keep = list(structured_data[0].keys())[:5]
                structured_data = [{k: row[k] for k in keys_to_keep if k in row} for row in structured_data]
    except Exception as e:
        logger.warning(f"Data truncation failed: {e}")

    prompt = f"""
    You are a helpful assistant. Summarize the result below in a user-friendly format.
    
    - Assume the user is a business owner looking for insights into their real estate portfolio.
    - Consider the data that I have is about the properties that were sold.
    - Do NOT say "assumed", "not available", or "not provided" if data like actual prices, dates, or city names are clearly present.
    - Don't give italicized text, just plain text.

    Question:
    {user_question}

    Data:
    {json.dumps(structured_data, indent=2)}

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
        logger.exception(f"Failed to generate answer: {e}")
        return "There was an error while generating the final answer."

@traceable(name="handle_routed_query")
def handle_routed_query(user_question, chat_memory_dict=None):
    logger.info(f"Handling new question: {user_question}")
    logger.info(f"Chat memory provided: {chat_memory_dict}")

    # Step 0: Check for general message
    if is_general_message(user_question):
        return generate_friendly_reply(user_question)

    def run_query(query, memory=None):
        logger.info("Invoking routing logic...")
        route_info = route_query_with_function_call(query, chat_memory_dict=memory)
        destination = route_info.get("destination")
        property_mention = route_info.get("property_mention")
        agent_mention = route_info.get("agent_mention")

        logger.info(f"Route: {destination}")
        logger.info(f"Property Mention: {property_mention}")
        logger.info(f"Agent Mention: {agent_mention}")

        try:
            if destination == "sql":
                logger.info("Routing to SQL backend...")
                conn = get_sqlite_conn()
                result = handle_user_question(query, conn, property_mention, agent_mention, memory)
                logger.info(f"SQL result: {result}")
                return result, destination

            elif destination == "firestore":
                logger.info("Routing to Firestore backend...")
                result = handle_user_feedback_query(
                    user_question=query,
                    collection_name="feedback_feedback",
                    property_mention=property_mention,
                    agent_mention=agent_mention,
                    chat_memory_dict=memory
                )
                logger.info(f"Firestore result: {result}")
                return result, destination

            elif destination == "faiss":
                logger.info("Routing to FAISS backend...")
                result = search_property_pdfs(faiss_index, query, property_mention, chat_memory_dict=memory)
                if "error" in result or not result.get("context"):
                    logger.warning("FAISS returned no results or context was empty.")
                    return None, destination
                logger.info("Passing FAISS context to LLM for final answer...")
                return generate_answer_from_context(query, result["context"], client), destination

            logger.warning("Destination was unrecognized.")
            return "Invalid routing destination.", "invalid"

        except Exception as e:
            logger.exception(f"Query error during handler execution: {e}")
            return "An error occurred while processing your query.", "error"

    # Primary run
    result, destination = run_query(user_question, memory=chat_memory_dict)

    if result and (not isinstance(result, str) or "no relevant" not in result.lower()):
        return generate_natural_answer(user_question, result) if isinstance(result, list) else result

    # Fallback with chat memory
    if chat_memory_dict:
        logger.info("Trying fallback with full chat history...")
        fallback_result, _ = run_query(user_question, memory=chat_memory_dict)
        if fallback_result:
            return generate_natural_answer(user_question, fallback_result) if isinstance(fallback_result, list) else fallback_result

    return f"Sorry, no relevant information found for: **{user_question}**."
