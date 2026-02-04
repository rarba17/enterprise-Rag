import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA

load_dotenv()

# Connection string for the vector store (sync driver)
CONNECTION_STRING = os.getenv(
    "SYNC_DATABASE_URL",
    "postgresql+psycopg2://postgres:password@localhost:5433/rag_db"
)


def ask_question(question: str):
    embeddings = OpenAIEmbeddings()

    # 1. Connect to existing vector store in Postgres
    vectorstore = PGVector.from_existing_index(
        embedding=embeddings,
        collection_name="enterprise_docs",
        connection_string=CONNECTION_STRING,
    )

    # 2. Setup the LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 3. Create the Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # 4. Get the answer
    response = qa_chain.invoke(question)
    return {"question": question, "answer": response["result"]}