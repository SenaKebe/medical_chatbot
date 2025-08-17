import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config


def load_pdf(file_path):
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_and_process_documents():
    """Load and process PDFs into a FAISS vector store with optimizations"""
    try:
        start_time = time.time()

        # Check if FAISS index files exist
        index_faiss = os.path.join(Config.FAISS_INDEX_PATH, "index.faiss")
        index_pkl = os.path.join(Config.FAISS_INDEX_PATH, "index.pkl")

        if os.path.exists(index_faiss) and os.path.exists(index_pkl):
            print("Loading existing FAISS index...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': False}
            )
            vector_store = FAISS.load_local(
                Config.FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded existing vector store in {time.time() - start_time:.2f} seconds")
            return vector_store

        # Otherwise, create a new FAISS index
        if not os.path.exists(Config.DATA_PATH):
            os.makedirs(Config.DATA_PATH, exist_ok=True)
            print(f"Created {Config.DATA_PATH}. Please add PDF files and rerun.")
            return None

        pdf_files = list(Path(Config.DATA_PATH).glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {Config.DATA_PATH}")
            return None

        print(f"Loading {len(pdf_files)} PDF files in parallel...")
        load_start = time.time()
        documents = []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_pdf, pdf_files))
            for result in results:
                documents.extend(result)
        print(f"Loading took {time.time() - load_start:.2f} seconds")

        if not documents:
            print(f"No documents loaded from {Config.DATA_PATH}")
            return None

        print(f"Loaded {len(documents)} documents. Splitting into chunks...")
        split_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Splitting took {time.time() - split_start:.2f} seconds")

        print(f"Creating embeddings for {len(chunks)} chunks...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )

        print("Building FAISS vector store...")
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Save the FAISS index
        vector_store.save_local(Config.FAISS_INDEX_PATH)

        print(f"✅ Vector database created and saved successfully in {time.time() - start_time:.2f} seconds")
        return vector_store

    except Exception as e:
        print(f"❌ Error during document processing: {e}")
        return None
