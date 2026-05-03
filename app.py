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
from langchain.schema.runnable import RunnableLambda

load_dotenv()

app = Flask(__name__)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GROQ_API_KEY']=GROQ_API_KEY

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

client = Groq(api_key=GROQ_API_KEY)

def generate_answer(input):
    prompt_text = str(input)
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text }
        ]
    )
    return response.choices[0].message.content

llm = RunnableLambda(generate_answer)

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = retrieval_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080, debug = True)