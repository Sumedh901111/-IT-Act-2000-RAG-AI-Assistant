import streamlit as st
import pickle
from transformers import pipeline

#load FAISS index from pickle
with open("indexes/vectore_store.pkl","rb") as f:
    vector_store=pickle.load(f)

#load local huggingface model 
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

st.title("📘 Legal AI Assistant (IT Act, 2000)")
st.write("Ask questions about the Indian IT Act, 2000")
query = st.text_input("Enter your question:")

if query:
    # Retrieve top chunks
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # Format prompt
    prompt = f"Answer based on the IT Act, 2000:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    # Generate answer
    answer = qa_model(prompt, max_length=200, do_sample=False)[0]['generated_text']

    st.subheader("Answer:")
    st.write(answer)

    # Show retrieved context
    with st.expander("Show retrieved context"):
        st.write(context)