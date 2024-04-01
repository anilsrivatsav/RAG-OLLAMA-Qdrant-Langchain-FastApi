from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import uvicorn
import logging
import os
from tqdm import tqdm
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models

class Queryy(BaseModel):
    query: str
    top_k: int = 5

logging.basicConfig(level=logging.INFO)
global_retriever = None
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
embeddings = OllamaEmbeddings(model="nomic-embed-text")

app = FastAPI()


def process_and_store_documents(documents, collection_name="my_documents", batch_size=100):
    global global_retriever

    if documents:
        start_time = time.time()
        if global_retriever is None:
            # Initialize the Qdrant vector store if it's not already initialized
            qdrant = Qdrant.from_documents(
                documents,
                embeddings,
                url="localhost:6333",
                collection_name=collection_name,
                force_recreate=True,
            )
            global_retriever = qdrant.as_retriever()
        else:
            # Retrieve existing document IDs from the Qdrant vector store
            existing_ids = set([doc.id for doc in global_retriever.vector_store.get_all_documents()])

            # Process and store new documents in batches with progress tracking
            total_added = 0
            for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
                batch = documents[i:i + batch_size]
                new_documents = [doc for doc in batch if doc.id not in existing_ids]
                if new_documents:
                    global_retriever.add_documents(new_documents)
                    total_added += len(new_documents)
                    logging.info(f"Added {len(new_documents)} new documents to Qdrant collection: {collection_name}")
                    existing_ids.update([doc.id for doc in new_documents])

            if total_added == 0:
                logging.info("No new documents to add to Qdrant collection.")

        end_time = time.time()
        logging.info(f"Processing completed in {end_time - start_time:.2f} seconds.")

# Process the files from inside data/
pdf_loader = DirectoryLoader("db/", glob="**/*.pdf", loader_cls=PyPDFLoader)
loaded_documents = pdf_loader.load()

# Split loaded documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
docs1 = text_splitter.split_documents(loaded_documents)

# Store the document chunks from the db/ directory in the Qdrant vector store
process_and_store_documents(docs1, collection_name="my_documents")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"{ABS_PATH}/data/{file.filename}"
    try:
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    # Process the uploaded file
    docs = text_splitter.split_documents([file_location])

    # Check if documents already exist in the collection
    if global_retriever is not None:
        existing_ids = [doc.id for doc in global_retriever.vector_store.get_all_documents()]
        new_documents = [doc for doc in docs if doc.id not in existing_ids]

        # Store only new document chunks in the Qdrant vector store
        if new_documents:
            process_and_store_documents(new_documents, collection_name="my_documents")
            return {"status": "success", "message": "New file uploaded and processed successfully"}
        else:
            return {"status": "warning", "message": "No new documents found in the uploaded file"}
    else:
        return {"status": "error", "message": "Retriever not initialized. Please initialize Qdrant first."}

@app.get("/retrieve/")
def retrieve_top_contexts(query_data: Queryy):
    query = query_data.query
    top_k = query_data.top_k
    if global_retriever is None:
        return {"status": "error", "message": "Retriever not initialized. Please upload documents first."}

    top_contexts = global_retriever.get_relevant_documents(query, top_k=top_k)
    return {
        "status": "success",
        "query": query,
        "top_contexts": top_contexts
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
