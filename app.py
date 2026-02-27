import os
# Enable MPS for Mac acceleration
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.vector_db import vector_db

st.set_page_config(page_title="Educational LLM Chat", layout="wide")

st.title("🎓 Private Educational LLM")
st.markdown("Interact with your small, custom-trained LLM.")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "models/educational-qwen-0.5b-lora"

@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading base model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    
    adapter_config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"Loading adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        st.sidebar.success("✅ Custom educational adapter loaded!")
    else:
        st.sidebar.warning("⚠️ Specialized adapter not ready yet. Using base model.")
        
    return model, tokenizer

model, tokenizer = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Retrieve context from Vector DB for RAG
        context_list = vector_db.query(prompt)
        context_text = "\n".join(context_list) if context_list else "No additional context found."
        
        # Format for Qwen Chat Specialist with Context
        input_text = f"<|im_start|>system\nYou are an expert educational mathematics tutor. Use the provided context to help answer accurately. Always use LaTeX for mathematical notation (e.g., use $x^2$ for exponentiation). Provide clear, step-by-step professional explanations.\n\nContext:\n{context_text}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        device = next(model.parameters()).device
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
        end_time = time.time()
        time_taken = end_time - start_time
        
        full_response = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        message_placeholder.markdown(f"{full_response}\n\n*⏱️ Time taken: {time_taken:.2f}s*")
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.title("Model Stats")
st.sidebar.info(f"Base: {MODEL_ID}")
if os.path.exists("data/processed.jsonl"):
    with open("data/processed.jsonl", "r") as f:
        num_examples = len(f.readlines())
    st.sidebar.write(f"Training Examples: {num_examples}")
