import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Connection string for the vector store (sync driver)
CONNECTION_STRING = os.getenv(
    "SYNC_DATABASE_URL",
    "postgresql+psycopg2://postgres:password@localhost:5433/rag_db"
)


def get_vectorstore():
    """Get the vector store instance."""
    embeddings = OpenAIEmbeddings()
    return PGVector.from_existing_index(
        embedding=embeddings,
        collection_name="enterprise_docs",
        connection_string=CONNECTION_STRING,
    )


def debug_retrieval(question: str):
    """Debug function to see what chunks are retrieved."""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search_with_score(question, k=10)
    return [
        {"content": doc.page_content, "score": score, "metadata": doc.metadata}
        for doc, score in docs
    ]


PROMPT_TEMPLATE = """Use the following pieces of context to answer the question.
If you find the answer in the context, provide it directly and precisely.
If the answer is not in the context, say "I couldn't find this information in the documents."

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)



def ask_question(question: str):
    vectorstore = get_vectorstore()

    #  Setting up the LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 2. Create retriever for better settings
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
        }
    )

    # 3. creating a chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # 4. Get the answer
    response = qa_chain.invoke(question)
    return {"question": question, "answer": response["result"]}


async def ask_question_stream(question: str) -> AsyncGenerator[str, None]:
    """Stream the answer token by token."""
    vectorstore = get_vectorstore()

    # 1. Retrieve relevant documents
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )
    docs = retriever.invoke(question)

    # 2. Build context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Setup streaming LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

    # 4. Create messages with context
    messages = [
        SystemMessage(content=f"""Use the following context to answer the question.
If you find the answer in the context, provide it directly and precisely.
If the answer is not in the context, say "I couldn't find this information in the documents."

Context:
{context}"""),
        HumanMessage(content=question)
    ]

    # 5. Stream the response
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content