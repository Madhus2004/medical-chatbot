from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("templates/index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080, debug = True)