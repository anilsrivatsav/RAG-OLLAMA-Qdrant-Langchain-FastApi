# from fastapi import FastAPI, HTTPException, Query
# from pydantic import BaseModel
# from langchain_groq import ChatGroq
# from ingest import retrieve_top_contexts, Queryy
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os
# from langchain_core.output_parsers import StrOutputParser
# from ingest import global_retriever

# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")

# app = FastAPI()

# # LLM models
# chat_groq_model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# ollama_model = ChatOllama(model="mistral")

# class QARequest(BaseModel):
#     question: str
#     model: str = Query(default="groq", regex="^(groq|ollama)$")
#     top_k: int = 5
#     custom_prompt: str = ""

# @app.post("/qa/")
# async def qa(request: QARequest):
#     # Select the LLM model
#     if request.model == "groq":
#         llm = chat_groq_model
#     elif request.model == "ollama":
#         llm = ollama_model

#     # Set the prompt
#     prompt_template = PromptTemplate(
#         input_variables=["context", "question"],
#         template="{context}\nQuestion: {question}\nAnswer:",
#     )
    
#     query_data = Queryy(query=request.question, top_k=request.top_k)

#     # Retrieve top contexts using the imported function
#     top_contexts_response = retrieve_top_contexts(query_data)
    
#     if top_contexts_response["status"] == "error":
#         raise HTTPException(status_code=500, detail=top_contexts_response["message"])

#     top_contexts = [doc.page_content for doc in top_contexts_response["top_contexts"]]
#     # Form the QA chain using LCEL
#     prompt = prompt_template.format_prompt(context='\n'.join(top_contexts), question=request.question)
#     retrieval_chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

#     # Invoke the QA chain
#     res = retrieval_chain.invoke(request.question)
#     answer = res["result"]
#     source_documents = res["source_documents"]

#     response = {"answer": answer}

#     if source_documents:
#         response["sources"] = [source_doc.page_content for source_doc in source_documents]

#     return response

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8018)


# from fastapi import FastAPI
# # from langchain_groq import ChatGroq
# from langchain_community.chat_models import ChatOllama
# from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.messages import get_buffer_string
# from langchain.memory import ConversationBufferMemory
# from ingest import retrieve_top_contexts, Queryy, global_retriever
# from dotenv import load_dotenv
# from langchain_core.prompts import format_document
# import os

# load_dotenv()

# app = FastAPI()

# # LLM models
# # chat_groq_model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# ollama_model = ChatOllama(model="mistral")

# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

# def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
#     doc_strings = [format_document(doc, document_prompt) for doc in docs]
#     return document_separator.join(doc_strings)

# from operator import itemgetter

# memory = ConversationBufferMemory(
#     return_messages=True, output_key="answer", input_key="question"
# )

# # First we add a step to load memory
# loaded_memory = RunnablePassthrough.assign(
#     chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
# )
# # Now we calculate the standalone question
# standalone_question = {
#     "standalone_question": {
#         "question": lambda x: x["question"],
#         "chat_history": lambda x: get_buffer_string(x["chat_history"]),
#     }
#     | CONDENSE_QUESTION_PROMPT
#     | ollama_model
#     | StrOutputParser(),
# }
# # Now we retrieve the documents
# retrieved_documents = {
#     "docs": itemgetter("standalone_question") | global_retriever,
#     "question": lambda x: x["standalone_question"],
# }
# # Now we construct the inputs for the final prompt
# final_inputs = {
#     "context": lambda x: _combine_documents(x["docs"]),
#     "question": itemgetter("question"),
# }
# # And finally, we do the part that returns the answers
# answer = {
#     "answer": final_inputs | ANSWER_PROMPT | ollama_model,
#     "docs": itemgetter("docs"),
# }
# # And now we put it all together!
# final_chain = loaded_memory | standalone_question | retrieved_documents | answer

# inputs = {"question": "give 5 fitness tips"}
# result = final_chain.invoke(inputs)

# # For now you need to save it yourself
# memory.save_context(inputs, {"answer": result["answer"].content})

# memory.load_memory_variables({})

# inputs = {"question": "5 more"}
# result = final_chain.invoke(inputs)

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

ollama_host = "10.77.48.165:11434"
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
