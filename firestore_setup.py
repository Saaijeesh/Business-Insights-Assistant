import json
import os
from google.cloud import firestore
from dotenv import load_dotenv
from langsmith import traceable  

# @traceable will log this function execution
@traceable(name="upload_feedback_jsons_to_firestore")
def upload_feedback_jsons_to_firestore(
    feedback_dir="data/Feedback",
    project_id="finance-project-460719",
    database_id="nosqldb"
):
    load_dotenv()

    # Initialize Firestore client
    db = firestore.Client(project=project_id, database=database_id)

    # Iterate through all JSON files in the directory
    for filename in os.listdir(feedback_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(feedback_dir, filename)

            with open(json_path, "r") as f:
                feedback_list = json.load(f)

            base_name = os.path.splitext(filename)[0]  
            collection_name = f"{base_name}_feedback"

            for feedback in feedback_list:
                doc_id = f"agent_{feedback['agent_id']}_property_{feedback['property_id']}"
                db.collection(collection_name).document(doc_id).set(feedback)

    return db
