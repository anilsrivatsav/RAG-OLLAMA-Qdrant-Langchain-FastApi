# RAG-OLLAMA-Qdrant-Langchain-FastApi


This project provides a FastAPI service for question answering using the ChatOllama model from the LangChain library. It retrieves relevant documents based on the query and uses the ChatOllama model to generate an answer.

## Prerequisites

Before running this service, you need to have the following installed and running:

- **Ollama:** The ChatOllama model should be installed and running locally. For installation instructions, refer to the [Ollama documentation](https://github.com/langchain/ollama).
- **Qdrant:** The Qdrant vector database should be hosted on a local server and exposed on port 6333. For installation and setup instructions, refer to the [Qdrant documentation](https://github.com/qdrant/qdrant).
Store the document chunks from the db/ directory in the Qdrant vector store.

## Features

- **Question Answering:** Given a question, the service retrieves relevant documents and uses the ChatOllama model to generate an answer.
- **Memory Integration:** The service maintains a conversation history to provide context for follow-up questions.

## Installation

### Using Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/anilsrivatsav/RAG-OLLAMA-Qdrant-Langchain-FastApi.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repository
   ```
3. Build the Docker image:
   ```bash
   docker build -t fastapi-qa-service .
   ```
4. Run the Docker container:
   ```bash
   docker run -d --name fastapi-qa-service -p 8000:8000 fastapi-qa-service
   ```

The service will be available at `http://localhost:8000`.

### Without Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/anilsrivatsav/RAG-OLLAMA-Qdrant-Langchain-FastApi.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repository
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Using Docker

The FastAPI server will start automatically when the Docker container is run. You can access the service at `http://localhost:8000`.

### Without Docker

To start the FastAPI server, run the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`.

### Endpoints

- `POST /query/`: Process a question and return an answer.

  Request body:
  ```json
  {
      "question": "Your question here"
  }
  ```

  Response:
  ```json
  {
      "answer": "Answer to your question"
  }
  ```

## Configuration

Configuration
The service can be configured using environment variables. Rename the env.example file to .env and update it with the following contents:

```makefile
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="rag"
OLLAMA_HOST="localhost:11434"
``` 

Replace the `LANGCHAIN_API_KEY` and `OLLAMA_HOST` values with your actual API key and Ollama host address, respectively.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If you have any further modifications or need additional assistance, feel free to let me know!