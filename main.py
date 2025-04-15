from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- CONFIG ---
DOC_PATH = "new.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "vector"
PERSIST_DIRECTORY = "./chroma_db"

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# --- CORS setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (You can specify specific domains here)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- LangChain Setup ---

def ingest_pdf(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        return data
    else:
        logging.error(f"PDF not found at {doc_path}")
        return None

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    return splitter.split_documents(documents)

def load_vector_db():
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(PERSIST_DIRECTORY):
        return Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
    else:
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None
        chunks = split_documents(data)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        return vector_db

def create_retriever(vector_db, llm):
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Rewrite the question in 5 different ways to help search a vector DB.
Original question: {question}"""
    )
    return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=prompt_template)

def create_chain():
    llm = ChatOllama(model=MODEL_NAME)
    vector_db = load_vector_db()
    if vector_db is None:
        raise RuntimeError("Failed to load vector DB.")
    retriever = create_retriever(vector_db, llm)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the following context:
{context}
Question: {question}"""
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Initialize at startup
chain = create_chain()

# --- API Models ---

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- Routes ---

@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = chain.invoke(input=question)
        return {"answer": response}
    except Exception as e:
        logging.error(f"Error during chain invoke: {e}")
        raise HTTPException(status_code=500, detail="Internal error during response generation")
