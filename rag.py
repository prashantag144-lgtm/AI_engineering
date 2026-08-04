from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

file_path = input("ENTER THE FILE PATH : ").strip()
path = Path(file_path)

loader = PyPDFLoader(path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

text = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = Chroma.from_documents(
    documents=text,
    embedding=embedding,
)

retriever = vector.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 1000,
        "lambda_mult": 0.5,
    },
)

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)
prompt = ChatPromptTemplate.from_template(
    """Act as a resume analyzer. Use the resume context below to answer the user's question accurately.

Context:
{context}

Question:
{question}"""
)

question = input("enter your Query : ").strip()

retrieved_docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in retrieved_docs if getattr(doc, "page_content", None))

prompt_value = prompt.invoke({
    "question": question,
    "context": context,
})

response = llm.invoke(prompt_value)
print(response.content)