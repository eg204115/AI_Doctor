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
from datetime import datetime

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

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    users_collection = db['users']
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # Fallback to a simple dictionary for user storage (for demo purposes only)
    users_collection = None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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
        return render_template('chat.html')
    else:
        return redirect(url_for('login'))

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
                # Update last login time
                users_collection.update_one(
                    {"_id": user['_id']},
                    {"$set": {"last_login": datetime.utcnow()}}
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
            
            # Create new user
            new_user = {
                "username": username,
                "password": hashed_password,
                "email": email,
                "created_at": datetime.utcnow(),
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

@app.route("/get", methods=["GET", "POST"])
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)