import streamlit as st
import pickle
from transformers import pipeline

# ------------------------------
# Load FAISS index for IT Act, 2000
# ------------------------------
with open("indexes/vector_store.pkl", "rb") as f:
    vector_store = pickle.load(f)

# ------------------------------
# Load local HuggingFace model
# ------------------------------
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="⚖️ IT Act AI Assistant", layout="wide")
st.title("⚖️ IT Act, 2000 AI Assistant (Chat Style)")
st.write("Ask questions about the IT Act, 2000. Previous Q&A will be saved in search history.")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: clear history
st.sidebar.title("Settings")
if st.sidebar.button("Clear Chat History"):
    st.session_state.history = []
    st.success("Chat history cleared! ✅")

# User input
query = st.text_input("Enter your question:")

if query:
    # Retrieve top 3 chunks automatically
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # Generate answer with LLM
    prompt = f"Answer this question concisely using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = qa_model(prompt, max_length=200, do_sample=False)[0]["generated_text"]

    # Save to chat history
    st.session_state.history.append({"question": query, "answer": answer})

# Display chat history
st.subheader("📜 Search History")
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**AI:** {chat['answer']}")
    st.markdown("---")
