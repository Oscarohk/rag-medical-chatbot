from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings_model
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

os.chdir("../")
extracted_data = load_pdf_files(data="./data")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)
embeddings_model = download_embeddings_model()


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medibot" # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,      # Dimension of the embeddings
        metric="cosine",    # Cosine Similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings_model,
    index_name=index_name
)