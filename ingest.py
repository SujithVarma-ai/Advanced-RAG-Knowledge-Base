import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv

MODEL = "gemini-2.5-flash-lite"

DB_NAME = str(Path(__file__).parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent / "knowledge-base")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv(override=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks
# Suppose one document has 2000 characters.
# With:

#chunk_size = 500
#chunk_overlap = 200

#LangChain creates chunks like:
#Chunk 1 : 1 - 500
#Chunk 2 : 301 - 800
#Chunk 3 : 601 - 1100
#Chunk 4 : 901 - 1400
#Chunk 5 : 1201 - 1700
#Chunk 6 : 1501 - 2000

#Notice:
#Chunk 1 ends at 500
#Chunk 2 starts at 301

#The overlap is:
#301-500 = 200 characters

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    print("Documents:", len(documents))
    chunks = create_chunks(documents)
    print("Chunks:", len(chunks))
    create_embeddings(chunks)
    print("Ingestion complete")