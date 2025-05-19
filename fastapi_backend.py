import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler
from typing import List

# Path to your README folder
README_DIR = "readmes"

# Allow frontend access
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# List available README files
def get_readme_files():
    return [f for f in os.listdir(README_DIR) if f.endswith(".md")]

# Load selected README file
def load_readme(file_name):
    path = os.path.join(README_DIR, file_name)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return f.read()

# Custom callback handler to stream tokens
class WebSocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    def on_llm_new_token(self, token: str, **kwargs):
        # Send each token as it's generated
        import asyncio
        asyncio.create_task(self.websocket.send_text(token))

# Endpoint to list available README files
@app.get("/readme-files")
def list_readmes():
    return {"files": get_readme_files()}

# WebSocket for streaming responses
@app.websocket("/ws/ask")
async def ask_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            question = data["question"]
            readme_file = data["readme_file"]

            readme_content = load_readme(readme_file)
            if not readme_content:
                await websocket.send_text("Invalid README file selected.")
                continue

            prompt = (
                f"You are an assistant that answers questions based on the following README content:\n\n"
                f"```markdown\n{readme_content}\n```\n"
                f"Only answer questions based on the content. If out of scope, say so.\n\n"
                f"User question: {question}\nAnswer:"
            )

            # Stream response
            handler = WebSocketCallbackHandler(websocket)
            llm = OllamaLLM(model="llama3", streaming=True, callbacks=[handler])
            llm.invoke(prompt)  # Triggers streaming
            #print("Response: " + _)
            await websocket.send_text("[DONE]")  # signal end of stream

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
