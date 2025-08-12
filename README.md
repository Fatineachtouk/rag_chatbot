
## Overview
This is a RAG system designed to answer the user based on the text extracted via OCR. Technologies used:  
- **Chroma DB** for storing docs and their embeddings locally.  
- **BAAI/bge-m3** for multilingual embedding.  
- **LangChain** for embedding and chunking documents. The method used for chunking is **RecursiveCharacterTextSplitter**.  
- **Ollama** for pulling the LLM model used to generate the answer. The model used is **gemma2**.  

Brief explanation of the files:  
- **ragg.py**: All the functions used in storing, embedding, retrieving, and generating the answer.  
- **store_app.py**: The API (POST) that stores documents with their embeddings in Chroma DB.
- **delete_app.py**: The API that deletes a file(given the file_id) from the ChromaDB
- **chat.py**: The socket.  
- **evaluation**: Evaluating the model performance using ragas, here are the metrics used and the scores:  
  - Answer Correctness (using the right facts): 0.943  
  - Faithfulness (no hallucination): 1.000  
  - Answer Relevancy (on-topic answer): 0.795  
  - Context Recall (all needed info retrieved): 1.000  
  - Context Precision (only useful info retrieved): 0.786  
---

## How to Use Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Fatineachtouk/rag_chatbot.git
cd rag_chatbot
```

### 2. Set Up the Environment
First Make sure you have python 3.10 installed!
if not, you can install it here : [python 3.10](https://www.python.org/downloads/release/python-3100/)
<details> <summary> Windows</summary>
  
```bash
py -3.10 -m venv rag_env     # creating the environment
rag_env\Scripts\activate      # Activationg the environment
pip install -r requirements.txt
#Don't deactivate the env
```

</details> <details> <summary> macOS / Linux</summary>

```bash
python3.10 -m venv rag_env      # creating the environment
source rag_env/bin/activate       # Activationg the environment
pip install -r requirements.txt
#Don't deactivate the env
```
</details>

## Pulling ollama model
Make sure ollama is setup, if not, download it from the link : [download ollama](https://ollama.com/download), install it and then run the following in the terminal(preferably another terminal) :

```bash
ollama pull gemma2
```
## Setting up chroma DB
Create a folder for chroma locally, then replace the variable **CHROMA_DB_PATH** in .env file with the path of the new folder( make sure it is in the followig format : E:\\Projects\\RAG)

## Storing documents via the API
In the terminal, and while the environment is activated, run the following :
```python
python store_app.py
```
The server will be running on : http://127.0.0.1:5000/store-doc

### The input of the API :
```json
{
    "classification": "",
    "content": "",
    "metadata": [
        {
            "fieldName": "",
            "value": ""
        },
        {
            "fieldName": "",
            "value": ""
        }
    ],
    "file_id": ""
}
```
### The output :
```json
        {
            "status": "success",
            "chunks_stored":
        }
```
the `chunks_stored` is the number of chunks made out of the given document.

## Running the socket
In the terminal and while the environment is activated, run :
```bash
python chat.py
```
The socket will be running on port :5001,the full URL : http://localhost:5001

### Modifying the accessible File_ids
To change the file IDs that can be retrieved, go to the **chat.py** file. In line 8, you'll find the `file_ids` variable that you can modify.

## Deleting a file from the DB
In the activated terminal, run the following:
```bash
python delete_app.py
```
The API will be running on http://127.0.0.1:5050/delete_file
### The input of the API :
```json
{ "file_id": "000" }
```
### The output :
```json
{
    "deleted_ids": [
        "82cd714a-d068-4657-8734-25e005336aa2"
    ],
    "message": "Documents deleted successfully"
}
```

## Paramenters that can be modified
The modification of the following parameters can be done in the .env file:

- **CHUNK_SIZE**: The maximum length of characters per chunk.
- **CHUNK_OVERLAP**: The number of characters or tokens that are repeated between consecutive text chunks, so the model can understand the full meaning.
- **TOP_K**: The number of docs to retrieve in similarity search so that the LLM generates an answer based on them.
- **MAX_NEW_TOKENS**: Number of tokens the model will generate in response.
- **TEMPERATURE**: Controls randomness in text generation. The higher it is the more the answer is creative and random. In this case it's better to keep a low value for more confidence.
- **TOP_P**: Helps balance creativity and coherence, for e.g:if top_p=0.9, the model only picks from the top 90% probable next tokens, filtering out very unlikely words.
- **REPEAT_PENALTY**: Penalizes tokens that have already appeared in the generated text to reduce repetition.






