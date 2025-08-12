from flask import Flask, request, jsonify
from chromadb.utils import embedding_functions
from ragg import get_device, embedding, load_vectorstore
import os
app = Flask(__name__)

# Your Chroma DB path and embedding model
chroma_db_path = os.getenv("CHROMA_DB_PATH")
device = get_device()
embedding_model = embedding(device)


@app.route("/delete_file", methods=["POST"])
def delete_file():
    data = request.get_json()
    file_id = data.get("file_id")

    if not file_id:
        return jsonify({"error": "file_id is required"}), 400

    try:
        vectorstore = load_vectorstore(embedding_model)
        
        # Get all documents with the given file_id
        docs = vectorstore.get(where={"file_id": file_id})
        
        if not docs or len(docs["ids"]) == 0:
            return jsonify({"message": "No documents found with this file_id"}), 404

        # Delete by IDs
        vectorstore.delete(ids=docs["ids"])
        vectorstore.persist()

        return jsonify({
            "message": "Documents deleted successfully",
            "deleted_ids": docs["ids"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)
