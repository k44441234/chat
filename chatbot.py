# import streamlit as st
# from streamlit_chat import message
# import numpy as np
# import faiss
# import torch
# import json
# import os
# from openai import OpenAI
# from sentence_transformers import SentenceTransformer

# st.markdown("""
#     <style>
#     .bot-response {
#         color: red;
#         font-weight: bold;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# with open("data.json", encoding='utf-8') as f:
#     data = json.load(f)

# sentence = []
# for context in data:
#     sentence.append(context["content"])

# def chat_message(content, is_user=False, key=None):
#     alignment = "text-align: right;" if is_user else "text-align: right;"
#     direction = "direction: rtl;" if is_user else "direction: rtl;"
#     background_color = "#cccccc" if is_user else "#1b2331"
#     text_color = "color: #000000;"  if is_user else "color: #ffffff;"
#     st.markdown(
#         f"""
#         <div style="{direction} {alignment} {text_color} background-color: {background_color}; 
#                      border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; 
#                      border-radius: 10px;">
#             {content}
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# # Initialize session state
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
# if 'past' not in st.session_state:
#     st.session_state['past'] = []

# # 🧼 Add Clear Chat button
# if st.button("گفت و گو جدید"):
#     st.session_state['generated'] = []
#     st.session_state['past'] = []

# def get_text():
#     col1, col2 = st.columns([5, 0.5])
#     with col1:
#         input_text = st.text_input("", key="input")
#     with col2:
#         top_k = st.number_input("تعداد حدیث", min_value=1, max_value=20, value=5, step=1, label_visibility="visible")
#     return input_text, top_k


# st.markdown(
#     """
#     <h1 style='text-align: Right; direction: rtl;'>چت بات احادیث</h1>
#     """, 
#     unsafe_allow_html=True
# )

# search_text, top_k = get_text()

# model_name = "hamtaai/bg3_model".strip()
# model = SentenceTransformer(model_name, device="cpu")

# embedding_file = "embeddings.npy"

# if os.path.exists(embedding_file):
#     print("Loading existing embeddings from file...")
#     embeddings = np.load(embedding_file)
# else:
#     print("Embeddings file not found. Computing embeddings...")
#     embeddings = model.encode(sentence)
#     np.save(embedding_file, embeddings)
#     print("Embeddings saved to file.")

# vector_dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(vector_dimension)
# faiss.normalize_L2(embeddings)
# index.add(embeddings)
# print("Embeddings added to the FAISS index.")

# def final_text(query, top_k):
#     search_vector = model.encode(query)
#     _vector = np.array([search_vector])
#     faiss.normalize_L2(_vector)
#     distances, ann = index.search(_vector, top_k)
#     final_text = []
#     for item in ann[0]:
#         final_text.append(sentence[item])
#     return final_text

# client = OpenAI(
#     api_key=st.secrets["openai_key"],
# )

# def generate_answer(context, query):
#     prompt = f"احادیث: {context}\nسؤال: {query}"
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": """شما یک چت بات مختصص احادیث هست
#                 لطفا تنها با توجه به احادیث به سوال پاسخ بده"""
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )
#     return response.choices[0].message.content if response.choices else "No response received."

# def rag(query, top_k=5):
#     retrieved_docs = final_text(query, top_k)
#     combined_context = "\n حدیث: \n\n".join(retrieved_docs)
#     answer = generate_answer(combined_context, query)
#     return answer , query, combined_context  

# if search_text:
#     with st.spinner("Generating response..."):  
#         bot_response, context, combined_context  = rag(search_text, top_k)

#         response = bot_response
#         st.session_state.past.append(context)
#         st.session_state.generated.append(response)
#         with st.expander("📄 View retrieved context used for this answer"):
#             st.write(combined_context)

# if st.session_state['generated']:
#     for i in range(len(st.session_state['generated'])):
#         chat_message(st.session_state["generated"][i], is_user=False, key=str(i))









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
        body, .stApp {
            direction: rtl;
            text-align: right;
            font-family: "Tahoma", "Vazirmatn", sans-serif;
        }
        .bot-response {
            color: red;
            font-weight: bold;
        }
        .stTextInput > div > div > input {
            text-align: right !important;
        }
        .stNumberInput input {
            text-align: center;
        }
        h1, h2, h3, h4 {
            text-align: right !important;
        }
    </style>
    """, unsafe_allow_html=True)

with open("data.json", encoding='utf-8') as f:
    data = json.load(f)

sentence = [context["content"] for context in data]

def chat_message(content, is_user=False, key=None):
    background_color = "#cccccc" if is_user else "#1b2331"
    text_color = "#000000" if is_user else "#ffffff"
    st.markdown(
        f"""
        <div style="direction: rtl; text-align: right; color: {text_color}; 
                    background-color: {background_color}; 
                    border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; 
                    border-radius: 10px;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []


# پاک کردن چت و ورودی در صورت کلیک روی دکمه
if st.button("🔁 گفت‌وگوی جدید"):
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['input'] = ""  # پاک کردن ورودی
    st.experimental_rerun()         # ریفرش صفحه برای اعمال پاک‌سازی



def get_text():
    col1, col2 = st.columns([5, 1])
    with col1:
        input_text = st.text_input("پیام خود را وارد کنید:", key="input")
    with col2:
        top_k = st.number_input("تعداد احادیث", min_value=1, max_value=20, value=5, step=1, label_visibility="visible")
    return input_text, top_k


st.markdown(
    """
    <h1 style='text-align: right;'> چت‌بات پاسخگوی احادیث</h1>
    """,
    unsafe_allow_html=True
)

search_text, top_k = get_text()


model_name = "hamtaai/bg3_model".strip()
model = SentenceTransformer(model_name, device="cpu")

embedding_file = "embeddings.npy"
if os.path.exists(embedding_file):
    embeddings = np.load(embedding_file)
else:
    embeddings = model.encode(sentence)
    np.save(embedding_file, embeddings)

vector_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

def final_text(query, top_k):
    search_vector = model.encode(query)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    distances, ann = index.search(_vector, top_k)
    return [sentence[item] for item in ann[0]]

client = OpenAI(
    api_key=st.secrets["openai_key"],
)

def generate_answer(context, query):
    prompt = f"احادیث: {context}\nسؤال: {query}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """شما یک چت‌بات متخصص احادیث هستید.
                لطفاً فقط بر اساس محتوای احادیث پاسخ دهید."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content if response.choices else "پاسخی دریافت نشد."

def rag(query, top_k=5):
    retrieved_docs = final_text(query, top_k)
    combined_context = "\nحدیث:\n\n".join(retrieved_docs)
    answer = generate_answer(combined_context, query)
    return answer, query, combined_context

if search_text:
    with st.spinner("⏳ در حال تولید پاسخ..."):
        bot_response, context, combined_context = rag(search_text, top_k)
        st.session_state.past.append(context)
        st.session_state.generated.append(bot_response)
        with st.expander("📜 مشاهده احادیث بازیابی‌شده"):
            st.write(combined_context)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        chat_message(st.session_state["generated"][i], is_user=False, key=str(i))
