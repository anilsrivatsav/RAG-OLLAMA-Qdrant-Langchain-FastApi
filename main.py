from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import get_buffer_string
from langchain.memory import ConversationBufferMemory
from ingest import global_retriever
from langchain_core.prompts import format_document
from dotenv import load_dotenv
from operator import itemgetter

load_dotenv()
app = FastAPI()


ollama_model = ChatOllama(model="mistral", base_url=f"http://{ollama_host}")
condense_question_prompt = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

answer_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

default_document_prompt = PromptTemplate.from_template(template="{page_content}")

def combine_documents(docs, document_prompt=default_document_prompt, document_separator="\n\n"):
    return document_separator.join([format_document(doc, document_prompt) for doc in docs])

memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

loaded_memory = RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))

standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | condense_question_prompt
    | ollama_model
    | StrOutputParser(),
}

retrieved_documents = {
    "docs": itemgetter("standalone_question") | global_retriever,
    "question": lambda x: x["standalone_question"],
}

final_inputs = {
    "context": lambda x: combine_documents(x["docs"]),
    "question": itemgetter("question"),
}

answer = {
    "answer": final_inputs | answer_prompt | ollama_model,
    "docs": itemgetter("docs"),
}

final_chain = loaded_memory | standalone_question | retrieved_documents | answer

class Query(BaseModel):
    question: str

@app.post("/query/")
async def process_query(query: Query):
    inputs = {"question": query.question}
    try:
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"].content})
        return {"answer": result["answer"].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
