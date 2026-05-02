import os
from dotenv import load_dotenv
load_dotenv()
from src.helper import load_pdf_files, filter_minimal_doc, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GROQ_API_KEY']=GROQ_API_KEY

pinecone_api_key = PINECONE_API_KEY

extracted_data = load_pdf_files(data="./data")
minimal_docs = filter_minimal_doc(extracted_data)
text_chunks = text_split(minimal_docs)
embeddings = download_embeddings()
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric= "cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

stats=index.describe_index_stats()
vector_count=stats.get("total_vector_count", 0)

if vector_count == 0: 
    docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    index_name = index_name
    )
else:
    docsearch = PineconeVectorStore.from_existing_index(
    embedding = embeddings,
    index_name = index_name
)
