from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA

# Connection string for the vector store
CONNECTION_STRING = "postgresql+psycopg2://postgres:password@localhost:5432/rag_project"

@app.get("/ask")
async def ask_question(question: str):
    embeddings = OpenAIEmbeddings()

    # 1. Connect to our existing vector store in Postgres
    vectorstore = PGVector(
        collection_name="enterprise_docs",
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
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