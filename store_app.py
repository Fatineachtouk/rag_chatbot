from flask import Flask, request, jsonify
from ragg import ocr_to_chunks, store_to_chroma

app = Flask(__name__)

@app.route("/store-doc", methods=["POST"])
def store_document():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "content" not in data:
            return jsonify({"error": "Missing content field"}), 400
        
        if "file_id" not in data:
            return jsonify({"error": "Missing file_id field"}), 400
        
        # Pass the entire JSON to ocr_to_chunks (it will extract content and file_id)
        docs = ocr_to_chunks(data)
        
        if not docs:
            return jsonify({"error": "No chunks created"}), 400
        
        # Store to Chroma
        store_to_chroma(docs)
        
        return jsonify({
            "status": "success",
            "chunks_stored": len(docs)
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)