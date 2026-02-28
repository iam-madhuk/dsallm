import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import threading
from utils.dsa_scraper import scrape_dsa_content
from utils.dsa_preprocess import preprocess_dsa_data
from training.train import train as train_model
from utils.vector_db import VectorDB
import json
import time

app = FastAPI(title="DSA Specialist API", description="API for DSA Problem Solving LLM")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Separate adapter path for DSA
ADAPTER_PATH = "models/dsa-specialist-lora"

# Dedicated Vector DB for DSA
dsa_vector_db = VectorDB(db_path="data/dsa_chroma_db", collection_name="dsa_content")

model = None
tokenizer = None

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 768
    temperature: float = 0.5 # Lower temperature for better logic

class ChatResponse(BaseModel):
    response: str
    time_taken: float

@app.on_event("startup")
def load_model():
    global model, tokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading DSA Specialist model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    
    if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        print(f"Loading DSA adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        print("Note: DSA Adapter not found. Using base model with specialized prompting.")

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    context_list = dsa_vector_db.query(chat_request.prompt)
    context_text = "\n".join(context_list) if context_list else "No specific DSA context found."

    system_prompt = (
        "You are an expert Data Structures and Algorithms (DSA) specialist and technical interviewer. "
        "Your goal is to help the user solve coding problems optimally. "
        "1. Start with an explanation of the approach.\n"
        "2. Provide clean, well-commented Python code.\n"
        "3. Always include Time and Space complexity analysis using Big O notation.\n"
        "Use LaTeX for all mathematical notation and complexity ($O(N \\log N)$)."
    )

    input_text = f"<|im_start|>system\n{system_prompt}\n\nContext:\n{context_text}<|im_end|>\n<|im_start|>user\n{chat_request.prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=chat_request.max_tokens,
            do_sample=chat_request.temperature > 0,
            temperature=chat_request.temperature if chat_request.temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()
    
    response_text = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return ChatResponse(response=response_text, time_taken=round(end_time - start_time, 2))

@app.post("/train")
async def train_dsa(url: str):
    def run_pipeline():
        raw_file = "data/dsa_raw.txt"
        processed_file = "data/dsa_processed.jsonl"
        if scrape_dsa_content(url, raw_file):
            if preprocess_dsa_data(raw_file, processed_file) > 0:
                print("Training DSA model...")
                train_model(data_path=processed_file, num_epochs=1)
                # Note: In production we would need to reload the adapter here
    
    threading.Thread(target=run_pipeline).start()
    return {"message": "DSA training started", "url": url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
