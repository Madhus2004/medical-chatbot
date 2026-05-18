from typing import List
from langchain.schema import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.embeddings import Embeddings

import requests
import os


API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"
}


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


def filter_minimal_doc(docs: List[Document]) -> List[Document]:

    minimal_docs: List[Document] = []

    for doc in docs:

        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs


def text_split(minimal_docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )

    texts_chunks = text_splitter.split_documents(minimal_docs)

    return texts_chunks


class HFAPIEmbeddings(Embeddings):

    def embed_query(self, text):

        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": text,
                "options": {"wait_for_model": True}
            }
        )

        response.raise_for_status()

        embedding = response.json()

        return embedding[0]

    def embed_documents(self, texts):

        return [self.embed_query(text) for text in texts]


def download_embeddings():

    embeddings = HFAPIEmbeddings()

    return embeddings