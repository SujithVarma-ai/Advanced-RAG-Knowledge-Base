from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
DB_NAME = str(Path(__file__).parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
RETRIEVAL_K = 10

#User asks:
#Who won the IIOTY award?
#Chroma searches all:
#5000 chunks
#and returns:
#Top 10 chunks
#because:
#RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
print("DB_NAME =", DB_NAME)
print("Collection count =", vectorstore._collection.count())
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(temperature=0, model=MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question, history=[]):
    if isinstance(question, list):
        question = " ".join(map(str, question))
    prior = "\n".join(
        str(m["content"])
        for m in history
        if m["role"] == "user"
    )
    return prior + "\n" + str(question)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    print("Number of docs:", len(docs))

    for doc in docs[:2]:
        print(doc.metadata)
        print(doc.page_content[:200])
    return response.content, docs