from dotenv import load_dotenv
load_dotenv() 

from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

new_path=input("enter your path")
path=Path(new_path)
new_path=PyPDFLoader(path)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
    
)

text_split=splitter.split_documents(new_path)


embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectors=Chroma.from_documents(
    model=embeddings,
    chunks=splitter

)








