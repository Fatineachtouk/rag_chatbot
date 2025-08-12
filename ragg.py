from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import warnings
import logging
import ollama
import torch
import sys
import os


#loggings & warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.stdout.reconfigure(encoding='utf-8')

#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

#Environment Variables
load_dotenv()
chroma_db_path = os.getenv("CHROMA_DB_PATH")
model_name = os.getenv("MODEL_NAME")
top_k = int(os.getenv("TOP_K", 3))
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 80))
temperature = float(os.getenv("TEMPERATURE", 0.1))
top_p = float(os.getenv("TOP_P", 0.9))
penalty = float(os.getenv("REPEAT_PENALTY", 1.1))
chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP",200))


splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size, # at most
    chunk_overlap = chunk_overlap,
    separators=["\n\n", "\n", " ", ""]  # Tries to split on \n\n if not found, it tries with \n, if not foun with space..
)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def embedding(device):
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

device = get_device()
embedding_model = embedding(device)


def ocr_to_chunks(ocr_output):
    content = ocr_output.get("content", "")
    file_id = ocr_output.get("file_id", "")
    metadata_input = ocr_output.get("metadata", [])  # list of dicts
    
    if not content:
        return []
    
    meta_dict = {"file_id": file_id}
    #Creating a dictionary for metadata
    for item in metadata_input:
        field = item.get("fieldName")
        value = item.get("value")
        if field and value is not None:
            meta_dict[field] = value
    
    texts = splitter.split_text(content)
    return [Document(page_content=chunk, metadata=meta_dict) for chunk in texts]

def store_to_chroma(documents):
    # Create a persistent Chroma DB- Storing locally
    if not documents:
        return None
    
    try:#Assuming a DB alrady exists
        vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)#loads an existing database
        vectorstore.add_documents(documents)
    except:#Creating New Database
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=chroma_db_path)
    
    vectorstore.persist()
    return vectorstore


def load_vectorstore(embedding_model, persist_path=chroma_db_path):
    return Chroma(
        persist_directory=persist_path,
        embedding_function=embedding_model
    )


def detect_language(text):
    """Language detection"""
    if any(char in 'اأإآةىيءؤضصثقفغعهخحجدشسزرذنتالم' for char in text):
        return "ar"
    elif any(char in 'àâäçéèêëïîôùûüÿñæœ' for char in text.lower()):
        return "fr"
    elif any(char in 'ñáéíóúüç¿¡' for char in text.lower()):
        return "es"
    else:
        return "en"



def answer_generation_stream(query, accessible_file_ids):
    """Processes a user query and streams answer from the RAG pipeline."""
    vectorstore = load_vectorstore(embedding_model)

    if not query:
        yield "No question provided."
        return

    # Detect language
    lang = detect_language(query)

    # Retrieve relevant chunks while filtering 
    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k, filter={"file_id": {"$in": accessible_file_ids}}
        )
    except:
        try:#Creating the DB if not found
            docs = vectorstore.similarity_search(query, k=top_k, filter={"file_id": {"$in": accessible_file_ids}})
            results = [(doc, 1.0) for doc in docs]
        except:
            yield "Error retrieving documents."
            return

    if not results:
        yield "No relevant documents found."
        return

    # Extract metadata from results (file_id and all metadata fields)
    used_metadata = []
    for doc, score in results:
        doc_metadata = doc.metadata.copy()  # Get the entire metadata dictionary
        if doc_metadata not in used_metadata:
            used_metadata.append(doc_metadata)

    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Chunk {i+1}]\n{doc.page_content}"
        for i, (doc, score) in enumerate(results)
    ])

    # Build multilingual prompt (same as before)
    prompt = {
        "en": f"""You are a smart assistant. Answer the question based on the following context.
Answer in the same language as the question.

Context:
{context}

Question: {query}

Answer precisely:""",
        
        "fr": f"""Vous êtes un assistant intelligent. Répondez à la question en vous basant sur le contexte suivant.
Répondez dans la même langue que la question.

Contexte:
{context}

Question: {query}

Répondez précisément :""",
        
        "es": f"""Eres un asistente inteligente. Responde la pregunta basándote en el siguiente contexto.
Responde en el mismo idioma que la pregunta.

Contexto:
{context}

Pregunta: {query}

Responde con precisión:""",
        
        "ar": f"""أنت مساعد ذكي. أجب عن السؤال بناءً على السياق التالي.
أجب بنفس لغة السؤال.

السياق:
{context}

السؤال: {query}

أجب بدقة:"""
    }.get(lang, f"""You are a smart assistant. Answer the question based on the following context.
Answer in the same language as the question {lang}.

Context:
{context}

Question: {query}

Answer precisely:""")
    
    # Stream response from Ollama
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        stream=True,
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_new_tokens,
            "repeat_penalty": penalty
        }
    )

    for chunk in response:
        if chunk['response']:
            yield chunk['response']
    
    # Return used metadata as the last item
    yield used_metadata