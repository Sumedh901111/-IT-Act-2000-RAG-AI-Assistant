import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

#1. Load PDF
def load_pdf(path):
    reader=PdfReader(path)
    text=""
    for page in reader.pages:
        if page.extract_text():
            text+=page.extract_text() +"\n"
    return text
# Some PDF pages may have selectable text (like normal documents).
# Some PDFs may have scanned images (no actual text), in which case extract_text() returns None.
#2. Split text into chunks
def split_text(text):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)
#3. Create embeddings + FAISS index
def build_index(docs):
    embeddings=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-miniLm-L6-v2")
    vector_store=FAISS.from_texts(docs,embeddings)
    return vector_store

# PDF chunks -> [Embedding Model] -> Dense Vectors -> [FAISS Index]
# Query -> [Embedding Model] -> Query Vector -> Find nearest vectors -> Return relevant chunks

if __name__=="__main__":
    print("Loading PDF...")
    text=load_pdf("data/it act 2000.pdf")

    print("Splitting into chunks...")
    docs=split_text(text)

    print("Building Faiss index...")
    index=build_index(docs)

    #save to pickle
    with open("indexes/vectore_store.pkl","wb") as f:
        pickle.dump(index,f)

    print("Index built and saved to indexes/vector_store.pkl")