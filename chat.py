import os
import eventlet
import socketio
from flask import Flask
from ragg import answer_generation_stream, get_device, embedding

app = Flask(__name__)
sio = socketio.Server(cors_allowed_origins='*')
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

device = get_device()
embedding_model = embedding(device)

@sio.event
def connect(sid, environ):
   print(f"Client {sid} connected")
   sio.emit("status", {"message": "Connected to RAG server"}, to=sid)

@sio.event
def ask(sid, data):
   print(f"Received question from {sid}: {data}")
   
   question = data.get("question") or data.get("query", "")
   file_ids = data.get("file_ids", [])  # Get accessible file_ids from the request
   
   if not question.strip():
       sio.emit("stream_char", {"char": "No question provided."}, to=sid)
       return

   try:
       sio.emit("stream_start", {}, to=sid)
       
       tokens = list(answer_generation_stream(
           query=question,
           accessible_file_ids=["002", "005", "004"] # Accessible file ids
       ))
       
       # Last item is the metadata array
       used_metadata = tokens[-1] if isinstance(tokens[-1], list) else []
       
       # Print the used metadata
       print(f"Used metadata for this response: {used_metadata}")
   
       # Stream all tokens except the last one (which is metadata)
       for token in tokens[:-1]:
           sio.emit("stream_char", {"char": token}, to=sid)
           eventlet.sleep(0)
       
       sio.emit("stream_end", {"metadata": used_metadata}, to=sid)
       
   except Exception as e:
       error_msg = f"Error: {str(e)}"
       print(f"Error processing question: {e}")
       sio.emit("stream_char", {"char": error_msg}, to=sid)

@sio.event
def disconnect(sid):
   print(f"Client {sid} disconnected")

if __name__ == '__main__':
   eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5001)), app)