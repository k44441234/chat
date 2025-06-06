import streamlit as st
from streamlit_chat import message
import numpy as np
import faiss
import torch
import json
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer


st.markdown("""
    <style>
    .bot-response {
        color: red;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

with open("data.json", encoding='utf-8') as f:
    data = json.load(f)

sentence = []
for context in data[:10]:
    sentence.append(context["content"])


def chat_message(content, is_user=False, key=None):
    alignment = "text-align: left;" if is_user else "text-align: left;"
    direction = "direction: rlt;" if is_user else "direction: ltr;"
    background_color = "#cccccc" if is_user else "#1b2331"
    text_color = "color: #000000;"  if is_user else "color: #ffffff;"
    st.markdown(
        f"""
        <div style="{direction} {alignment} {text_color} background-color: {background_color}; 
                     border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; 
                     border-radius: 10px;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 🧼 Add Clear Chat button
if st.button("New Chat"):
    st.session_state['generated'] = []
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("", "", key="input")
    return input_text

st.markdown(
    """
    <h1 style='text-align: Left; direction: ltr;'>Book Guyton and Hall Textbook of Medical Physiology chatbot</h1>
    """, 
    unsafe_allow_html=True
)

search_text = get_text()
accuracy = []
model = SentenceTransformer("hamtaai/bg3_model", device="cpu")

embedding_file = "embeddings.npy"

if os.path.exists(embedding_file):
    print("Loading existing embeddings from file...")
    embeddings = np.load(embedding_file)
else:
    print("Embeddings file not found. Computing embeddings...")
    embeddings = model.encode(sentence)
    np.save(embedding_file, embeddings)
    print("Embeddings saved to file.")

vector_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print("Embeddings added to the FAISS index.")

def final_text(query, top_k):
    search_vector = model.encode(query)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    distances, ann = index.search(_vector, top_k)
    final_text = []
    for item in ann[0]:
        final_text.append(sentence[item])
    return final_text

client = OpenAI(
    api_key=st.secrets["openai_key"],
    # base_url="https://api.openai.com/v1"
)

def generate_answer(context, query):
    prompt = f"متن: {context}\nسؤال: {query}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a specialized medical assistant trained to help medical students understand complex physiological concepts from the Guyton and Hall Textbook of Medical Physiology."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content if response.choices else "No response received."

def rag(query, top_k=3):
    retrieved_docs = final_text(query, top_k)
    print(len(retrieved_docs))
    combined_context = "\n\n".join(retrieved_docs)
    answer = generate_answer(combined_context, query)
    return answer , query, combined_context  

if search_text:
    with st.spinner("Generating response..."):
        
        bot_response, context, combined_context  = rag(search_text)
        response = bot_response
        st.session_state.past.append(context)
        st.session_state.generated.append(response)
        # Show the combined context (retrieved paragraphs)
        with st.expander("📄 View retrieved context used for this answer"):
            st.write(combined_context)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        # chat_message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        chat_message(st.session_state["generated"][i], is_user=False, key=str(i))