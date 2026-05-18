from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from src.prompt import system_prompt
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pinecone_api_key = PINECONE_API_KEY
embeddings = download_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k":3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "Context: {context}\n\nQuestion: {input}")
    ]
)


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("✅ Retrieval chain ready")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    try:
        msg = request.form["msg"]

        msg_lower = msg.lower().strip()

        greetings = ["hi", "hello", "hey", "hii"]

        if msg_lower in greetings:
            return "Hi! How can I help you?"


        print("USER:", msg)

        print("STEP 1")

        response = retrieval_chain.invoke({"input": msg})

        print("STEP 2")

        print(response)

        return str(response["answer"])

    except Exception as e:
        print("ERROR:")
        print(e)
        return str(e)
    

port = int(os.environ.get("PORT",8080))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = port, debug = False)