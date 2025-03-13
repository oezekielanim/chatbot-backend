from flask import Flask, request, jsonify
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

# Load FAISS index
index = faiss.read_index("hr_policy_faiss.index")

# Load HR policy mapping
with open("hr_policy_mapping.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Store ongoing conversations for follow-up questions
user_conversations = {}

# Predefined small talk responses
small_talk_responses = {
    "hello": ["Hello! How can I assist you today?", "Hi there! Need help with something? 😊"],
    "hi": ["Hey! What can I do for you?", "Hi! How's your day going?"],
    "how are you": ["I'm just a bot, but I'm doing great! How about you?", "I'm here and ready to assist!"],
    "thank you": ["You're welcome! 😊", "Anytime! Let me know if you need more help."]
}

# Predefined follow-up questions
follow_up_questions = [
    "Could you clarify what you mean?",
    "Are you asking about an HR policy?",
    "Can you rephrase that? I want to assist you better."
]

# Function to classify user input
def classify_input(query):
    query_lower = query.lower().strip()
    
    # Check if it's small talk
    for key in small_talk_responses.keys():
        if key in query_lower:
            return "small_talk"
    
    return "hr_policy"

# Function to search FAISS
def search_faiss(query, top_k=1):
    query_embedding = np.array([embedding_model.encode(query)])
    distances, indices = index.search(query_embedding, top_k)

    # If the best match is too far away, assume it's not relevant
    if distances[0][0] > 1.0:  # Adjust this threshold if needed
        return None

    results = [documents[idx] for idx in indices[0]]
    return results[0][1]  # Return the best answer

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    user_id = data.get("user_id", "default_user")  # Track user sessions
    query = data.get("question", "").strip()

    if not query:
        return jsonify({"error": "No question provided"}), 400

    # Handle small talk
    category = classify_input(query)
    if category == "small_talk":
        response = random.choice(small_talk_responses[query.lower().strip()])
        return jsonify({"response": response})

    # If user was asked a follow-up, process their response
    if user_id in user_conversations and user_conversations[user_id].get("waiting_for_clarification"):
        # Try searching again with the new input
        answer = search_faiss(query)
        if answer:
            response = answer
        else:
            response = "I'm still unsure. Maybe try asking in a different way? 😊"
        user_conversations[user_id]["waiting_for_clarification"] = False  # Reset follow-up status
        return jsonify({"response": response})

    # Standard HR policy search
    answer = search_faiss(query)
    if answer:
        response = answer
    else:
        response = random.choice(follow_up_questions)  # Ask a follow-up
        user_conversations[user_id] = {"waiting_for_clarification": True}  # Mark as waiting for user clarification

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
