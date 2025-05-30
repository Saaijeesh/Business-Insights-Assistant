import os
import json
import logging
import sqlite3
import pandas as pd
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from sqlite_setup import get_sqlite_conn
from langsmith import traceable  

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Generate SQL from user query using GPT
@traceable(name="generate_sql_query")
def generate_sql_query(user_question, previous_query=None, error_msg=None):
    schema = """
    You are a helpful assistant that converts natural language into valid SQLite SQL queries.

    The database has a table called `real_estate` with this schema:
    - purchase_id (INTEGER): Unique purchase identifier for a transaction.
    - property_id (INTEGER): Identifier for an apartment complex or property.
    - house_id (INTEGER): Identifier for a specific house/unit within a property. House ID may repeat across different properties.
      To uniquely identify a house, use the combination of property_id and house_id.
    - date (TEXT): Date of purchase (format YYYY-MM-DD).
    - city (TEXT): City in Connecticut where the house is located.
    - agent_id (INTEGER): Identifier of the agent (values range across multiple agents; not sequential).
    - expected_sale_price (REAL): Expected price of the house, typically between 300,000 and 1,000,000.
    - actual_sale_price (REAL): Actual sold price, similar range as expected_sale_price.
    - number_of_days_on_listing (INTEGER): Days the house was listed before being sold.
    - number_of_beds (INTEGER): Number of bedrooms (2, 3, or 4).
    - number_of_baths (INTEGER): Number of bathrooms (2, 3, or 4).

    Your job is to:
    - Return a valid SQLite SQL query using this schema.
    - If the question includes phrases like "top N", "most", "least", "frequent", or "popular", use appropriate aggregation (e.g., COUNT) and sorting (e.g., ORDER BY COUNT DESC).
    - When calculating metrics like profit or prices across multiple houses, use `AVG()` or `SUM()` with aggregation
    - Always use aliases (e.g., AS house_count or AS most_common_city) for aggregated columns.
    - Do NOT include explanations, markdown, or comments. Only return the SQL query.
    """

    if error_msg and previous_query:
        logger.warning("[SQL] Previous query failed. Regenerating with error context.")
        user_question = (
            f"The following SQL query gave an error:\n\n{previous_query}\n\n"
            f"The error was:\n{error_msg}\n\n"
            f"Please fix and return a valid query using proper column aliases."
        )

    logger.info(f"[SQL] Generating SQL for: {user_question}")
    messages = [
        {"role": "system", "content": schema},
        {"role": "user", "content": user_question}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception(f"[SQL] Failed to generate SQL from GPT: {e}")
        return ""

# Determine if chat context is needed
@traceable(name="needs_chat_context")
def needs_chat_context(question, chat_memory_dict=None):
    if not chat_memory_dict:
        return False 

    chat_pairs = "\n".join([
        f"User: {chat_memory_dict[f'q{i}']}\nAssistant: {chat_memory_dict[f'a{i}']}"
        for i in range(1, len(chat_memory_dict) // 2 + 1)
    ])

    prompt = f"""
    You are an assistant that decides whether a user's current question depends on earlier conversation context.

    Prior Chat:
    {chat_pairs if chat_pairs else '[No prior chat]'}

    Current Question:
    {question}

    Answer with only one word: "yes" if it needs previous context, otherwise "no".
    """

    try:
        logger.info("[SQL] Determining if chat context is needed...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip().lower()
        return result == "yes"
    except Exception as e:
        logger.warning(f"[SQL] Context check failed: {e}")
        return False

# Traced SQL query handler with contextual memory
@traceable(name="handle_user_question")
def handle_user_question(user_question, conn, property_mention=None, agent_mention=None, chat_memory_dict=None, max_retries=3):
    attempt = 0
    error_msg = None
    sql_query = None

    logger.info(f"[SQL] Handling user question: {user_question}")

    if needs_chat_context(user_question, chat_memory_dict):
        logger.info("[SQL] Injecting chat memory into question...")
        memory_blocks = [
            f"User: {chat_memory_dict[f'q{i}']}\nAssistant: {chat_memory_dict[f'a{i}']}"
            for i in range(1, len(chat_memory_dict) // 2 + 1)
        ]
        memory_text = "\n".join(memory_blocks)
        user_question = f"Conversation so far:\n{memory_text}\n\nNow answer this: {user_question}"

    if property_mention:
        user_question = f"The user is referring to {property_mention}. " + user_question
    if agent_mention:
        user_question = f"The user is referring to {agent_mention}. " + user_question

    while attempt < max_retries:
        sql_query = generate_sql_query(user_question, sql_query, error_msg)
        if not sql_query:
            logger.error("[SQL] No SQL query was generated.")
            return "Failed to generate SQL."

        logger.info(f"[SQL] Attempt {attempt+1}: Generated query:\n{sql_query}")

        # Check for non-SQL responses like explanations or apologies
        if not sql_query.strip().lower().startswith(("select", "with")):
            logger.warning("[SQL] Detected non-SQL response from GPT.")
            return (
                "I can help you analyze your property data. "
                "I have access to sales data (CSV), customer and agent feedback (JSON), and property descriptions (PDFs). "
                "Let me know what insights you're looking for."
            )

        try:
            result_df = pd.read_sql_query(sql_query, conn)
            if not result_df.empty:
                logger.info(f"[SQL] Query successful. Rows returned: {len(result_df)}")
                return result_df.to_dict(orient='records')
            else:
                logger.warning("[SQL] Query executed but returned no rows.")
                return None
        except Exception:
            error_msg = traceback.format_exc()
            logger.warning(f"[SQL] Query failed on attempt {attempt+1}: {error_msg.splitlines()[-1]}")
            attempt += 1

    logger.error("[SQL] Query failed after maximum retries.")
    return None
