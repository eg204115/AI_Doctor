from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import hashlib
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timezone
import pytz

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_fallback_secret_key_here')

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# MongoDB connection
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.environ.get('DB_NAME', 'medical_chatbot')

# Initialize MongoDB variables
client = None
db = None
users_collection = None
chat_sessions_collection = None
messages_collection = None

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    users_collection = db['users']
    chat_sessions_collection = db['chat_sessions']
    messages_collection = db['messages']
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # Fallback to a simple dictionary for user storage (for demo purposes only)
    users_collection = None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_local_time():
    """Get current local time without timezone information"""
    return datetime.now()

embeddings = download_hugging_face_embeddings()

index_name = 'medicalbot-rag'

# Embed each chunk and upsert the embeddings into pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    if 'user_id' in session:
        # Get user's chat sessions
        user_id = session['user_id']
        chat_sessions = list(chat_sessions_collection.find(
            {"user_id": user_id}
        ).sort("last_activity", -1))
        
        # Get current session ID or create a new one
        current_session_id = session.get('current_session_id')
        if not current_session_id:
            # Create a new chat session
            new_session = {
                "user_id": user_id,
                "title": "New Chat",
                "created_at": get_local_time(),  # Use local time
                "last_activity": get_local_time(),  # Use local time
                "message_count": 0
            }
            result = chat_sessions_collection.insert_one(new_session)
            current_session_id = str(result.inserted_id)
            session['current_session_id'] = current_session_id
        
        # Get messages for current session
        messages = list(messages_collection.find(
            {"session_id": current_session_id}
        ).sort("timestamp", 1))
        
        return render_template('chat.html', 
                             chat_sessions=chat_sessions,
                             current_session_id=current_session_id,
                             messages=messages)
    else:
        return redirect(url_for('login'))

@app.route("/session/<session_id>")
def switch_session(session_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Verify the session belongs to the user
    session_obj = chat_sessions_collection.find_one({
        "_id": ObjectId(session_id),
        "user_id": session['user_id']
    })
    
    if session_obj:
        session['current_session_id'] = session_id
        flash('Switched to selected chat session', 'info')
    else:
        flash('Invalid chat session', 'error')
    
    return redirect(url_for('index'))

@app.route("/session/new")
def new_session():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Create a new chat session
    new_session = {
        "user_id": session['user_id'],
        "title": "New Chat",
        "created_at": get_local_time(),  # Use local time
        "last_activity": get_local_time(),  # Use local time
        "message_count": 0
    }
    result = chat_sessions_collection.insert_one(new_session)
    session['current_session_id'] = str(result.inserted_id)
    
    flash('Started a new chat session', 'info')
    return redirect(url_for('index'))

@app.route("/session/delete/<session_id>")
def delete_session(session_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Verify the session belongs to the user
    session_obj = chat_sessions_collection.find_one({
        "_id": ObjectId(session_id),
        "user_id": session['user_id']
    })
    
    if session_obj:
        # Delete the session and all its messages
        chat_sessions_collection.delete_one({"_id": ObjectId(session_id)})
        messages_collection.delete_many({"session_id": session_id})
        
        # If deleting current session, switch to a new one
        if session.get('current_session_id') == session_id:
            session.pop('current_session_id', None)
        
        flash('Chat session deleted', 'info')
    else:
        flash('Invalid chat session', 'error')
    
    return redirect(url_for('index'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = hash_password(password)
        
        if users_collection is not None:
            # MongoDB approach
            user = users_collection.find_one({
                "username": username, 
                "password": hashed_password
            })
            
            if user:
                session['user_id'] = str(user['_id'])
                session['username'] = user['username']
                # Clear any previous session ID
                session.pop('current_session_id', None)
                # Update last login time with local time
                users_collection.update_one(
                    {"_id": user['_id']},
                    {"$set": {"last_login": get_local_time()}}  # Use local time
                )
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error="Invalid credentials")
        else:
            # Fallback for demo purposes
            return render_template('login.html', error="Database not available")
    
    return render_template('login.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        email = request.form.get("email", "")
        hashed_password = hash_password(password)
        
        if users_collection is not None:
            # Check if username already exists
            if users_collection.find_one({"username": username}):
                return render_template('signup.html', error="Username already exists")
            
            # Create new user with local time
            new_user = {
                "username": username,
                "password": hashed_password,
                "email": email,
                "created_at": get_local_time(),  # Use local time
                "last_login": None  # No login yet
            }
            
            result = users_collection.insert_one(new_user)
            
            # CHANGED: Redirect to login page with success message instead of auto-login
            # Use flash message to show success
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            return render_template('signup.html', error="Database not available")
    
    return render_template('signup.html')

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route("/get", methods=["POST"])
def chat():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
        
    current_session_id = session.get('current_session_id')
    if not current_session_id:
        return jsonify({"error": "No active session"}), 400
        
    msg = request.form["msg"]
    input = msg
    print(input)
    
    # Save user message to database with local time
    user_message = {
        "session_id": current_session_id,
        "user_id": session['user_id'],
        "content": msg,
        "sender": "user",
        "timestamp": get_local_time()  # Use local time
    }
    messages_collection.insert_one(user_message)
    
    # Get AI response
    response = rag_chain.invoke({"input": msg})
    answer = str(response["answer"])
    print("Response : ", answer)
    
    # Save AI response to database with local time
    ai_message = {
        "session_id": current_session_id,
        "user_id": session['user_id'],
        "content": answer,
        "sender": "ai",
        "timestamp": get_local_time()  # Use local time
    }
    messages_collection.insert_one(ai_message)
    
    # Update session activity and message count with local time
    chat_sessions_collection.update_one(
        {"_id": ObjectId(current_session_id)},
        {
            "$set": {"last_activity": get_local_time()},  # Use local time
            "$inc": {"message_count": 2}  # Count both user and AI messages
        }
    )
    
    # Update session title if it's the first message
    session_obj = chat_sessions_collection.find_one({"_id": ObjectId(current_session_id)})
    if session_obj and session_obj.get('message_count', 0) <= 2:  # First exchange
        # Create a title from the first message (truncate if too long)
        title = msg[:30] + "..." if len(msg) > 30 else msg
        if not title.strip():
            title = "New Chat"
            
        chat_sessions_collection.update_one(
            {"_id": ObjectId(current_session_id)},
            {"$set": {"title": title}}
        )
    
    return answer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)