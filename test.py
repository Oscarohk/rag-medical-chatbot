import os
os.environ['HF_HOME'] = "E:/ohokin/.cache/"

from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings_model
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

embeddings_model = download_embeddings_model()

index_name = "medibot"
# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings_model
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
retrieved_docs = retriever.invoke("What is Acne?")
print(retrieved_docs)