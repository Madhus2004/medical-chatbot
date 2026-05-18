from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import requests
import os

HF_API_KEY = os.getenv("HF_API_KEY")
HF_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}


def _call_hf_api(texts):
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True}
    }
    response = requests.post(HF_URL, headers=HF_HEADERS, json=payload)
    response.raise_for_status()
    return response.json()


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def filter_minimal_doc(docs: List[Document]) -> List[Document]:
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source")}
        )
        for doc in docs
    ]


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(minimal_docs)


class HFAPIEmbeddings(Embeddings):

    def embed_query(self, text: str):
        print("CALLING HF EMBEDDING API (QUERY)", flush=True)
        result = _call_hf_api(text)
        if isinstance(result[0], list):
            return result[0]
        return result

    def embed_documents(self, texts: List[str]):
        print("CALLING HF EMBEDDING API (DOCUMENTS)", flush=True)
        all_embeddings = []
        for text in texts:
            result = _call_hf_api(text)
            if isinstance(result[0], list):
                all_embeddings.append(result[0])
            else:
                all_embeddings.append(result)
        return all_embeddings


def download_embeddings():
    return HFAPIEmbeddings()