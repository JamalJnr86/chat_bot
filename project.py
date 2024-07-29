import faiss
import numpy as np
import sqlite3
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import os
from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
api_key = os.getenv("api_key")


# Set your Groq API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Groq client and model
client = Groq(api_key=GROQ_API_KEY, base_url='https://api.groq.com')
model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# Function to load and chunk text from a .txt file
def load_and_chunk_text(file, chunk_size=1000):
    text = file.read().decode('utf-8')
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to load and chunk text from a PDF file
def load_and_chunk_pdf(file, chunk_size=1000):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to load and chunk text from a DOCX file
def load_and_chunk_docx(file, chunk_size=1000):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to index text chunks using FAISS
def index_chunks(chunks, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, model

# Function to retrieve relevant chunks using a query
def retrieve_chunks(query, index, model, chunks, top_k=5):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [chunks[i] for i in indices[0]]

# Function to generate a response using a language model
def generate_response(query, retrieved_chunks, generator_model_name='facebook/bart-large-cnn'):
    context = " ".join(retrieved_chunks)
    tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
    inputs = tokenizer(query + " " + context, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=200, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to generate a response using OpenAI API
def generate_openai_response(api_key, prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Initialize SQLite database
def init_db():
    with sqlite3.connect('chat_history.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        ''')
        # Clear the messages table
        # cursor.execute('DELETE FROM messages')
        conn.commit()

# Save messages to the database
def save_message(role, content):
    with sqlite3.connect('chat_history.db') as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO messages (role, content) VALUES (?, ?)', (role, content))
        conn.commit()

# Load messages from the database
def load_messages():
    with sqlite3.connect('chat_history.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT role, content FROM messages ORDER BY id ASC')
        return cursor.fetchall()

# Initialize database
init_db()

# Sidebar content
st.sidebar.title("BotBuddy Sidebar")

# Upload file in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

# Button to start a new chat
if st.sidebar.button("New Chat"):
    st.session_state.messages = []


st.sidebar.write("## Conversation History")

# Button to load previous history
if st.sidebar.button("Load Previous Chat"):
    st.session_state.messages = load_messages()
    st.sidebar.write("## Loaded Previous Chat History")
    for role, content in st.session_state.messages:
        role_display = "You" if role == "user" else "Assistant"
        st.sidebar.write(f"**{role_display}**: {content[:30]}{'...' if len(content) > 30 else ''}")


# Display chat messages from history in the sidebar
if "messages" in st.session_state:
    for role, content in st.session_state.messages:
        role_display = "You" if role == "user" else "Assistant"
        st.sidebar.write(f"**{role_display}**: {content[:30]}{'...' if len(content) > 30 else ''}")

st.markdown(
    """
    <h1 style='color: #FFD700;'>BotBuddy</h1>
    <p style='color: #FFD700;'>Your friendly conversational partner for all your queries and chats!</p>
    """,
    unsafe_allow_html=True
)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "meta-llama/Meta-Llama-3-70B-Instruct"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

# Display chat messages from history on app rerun
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# Handle file upload and query
if uploaded_file:
    # Determine the file type and load accordingly
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'txt':
        chunks = load_and_chunk_text(uploaded_file)
    elif file_type == 'pdf':
        chunks = load_and_chunk_pdf(uploaded_file)
    elif file_type == 'docx':
        chunks = load_and_chunk_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # Index the chunks for retrieval
    index, model = index_chunks(chunks)

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history and save to the database
        st.session_state.messages.append(("user", prompt))
        save_message("user", prompt)

        # Retrieve relevant chunks
        retrieved_chunks = retrieve_chunks(prompt, index, model, chunks)

        # Generate a response
        response = generate_response(prompt, retrieved_chunks)

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history and save to the database
        st.session_state.messages.append(("assistant", response))
        save_message("assistant", response)
else:
    # Accept user input when no file is uploaded
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history and save to the database
        st.session_state.messages.append(("user", prompt))
        save_message("user", prompt)

        # Generate a response using the OpenAI API
        response = generate_openai_response(api_key, prompt)

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history and save to the database
        st.session_state.messages.append(("assistant", response))
        save_message("assistant", response)
