# 🎓 Dsa LLM: Specialist Algorithm AI API

Dsa LLM is a custom-trained, small-scale Large Language Model (LLM) designed specifically for **DSA Problem Solving**. It leverages base Qwen models and is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) to provide expert-level support in Data Structures and Algorithms.

---

## � About the Project

**Dsa LLM** was born from the need for a focused, private, and high-performance coding tutor. While general-purpose LLMs are great, they often lack the deep algorithmic rigor required for technical interviews or specialized competitive programming.

This project focuses on:
- **Technical Rigor**: Providing not just code, but optimal approaches and complexity analysis.
- **Privacy-First**: No data leaves your machine. Inference and training happen locally.
- **Accessibility**: Using ultra-efficient 0.5B parameter models that run smoothly on average CPUs and Mac Silicon (MPS).
- **Customizability**: A built-in scraping and training pipeline allows anyone to "feed" the AI their own notes or curriculum.

---

## �🚀 Key Features

- **Algorithm Expert**: Built to be a Technical Interviewer, providing optimal code, approach explanations, and complexity analysis.
- **Contextual Memory (RAG)**: Uses **ChromaDB** with a specific `dsa_content` collection to store and retrieve algorithmic patterns.
- **REST API First**: Designed to be integrated into any coding platform or interview tool.
- **Speed Optimized**: Optimized for CPU/MPS (Metal Performance Shaders) execution for fast local inference.
- **LaTeX Complexity Output**: Automatically formats Big O notation using LaTeX (e.g., $O(N \log N)$).
- **End-to-End Pipeline**: Includes tools for web scraping and preprocessing DSA articles.

---

## 🏗 Technical Architecture

The project follows a modular design to separate data gathering, training, and real-time inference.

### 1. **The AI Brain (Qwen 0.5B)**
We use **Qwen2.5-0.5B-Instruct** as our base. It is powerful enough for logic but small enough to run on a standard laptop CPU without a high-end GPU. We apply **LoRA (Low-Rank Adaptation)** to specialize its responses for DSA interviews.

### 2. **Context Retrieval (RAG)**
Instead of relying solely on the model's pre-trained knowledge, we use **ChromaDB** as a vector database. When a user asks a question, the system retrieves relevant documents from the database to ensure the code and explanations are accurate and up-to-date.

---

## 🛠 File-by-File Breakdown

- **`api.py`**: The main FastAPI server. Loads the model and manages the chat/training endpoints.
- **`utils/vector_db.py`**: Handles all interactions with ChromaDB (indexing and querying).
- **`utils/dsa_scraper.py`**: Scrapes educational content from URLs (like GeeksforGeeks) to expand the AI's knowledge.
- **`utils/dsa_preprocess.py`**: Formats raw text into "ChatML" style pairs (Problem/Solution) for fine-tuning.
- **`training/train.py`**: The core script for fine-tuning the base model on your own local data.

---

## 🚦 Getting Started

### 1. Installation

Clone the repo and install the requirements (optimized for CPU/Mac):

```bash
pip install -r requirements.txt
```

### 2. Running the API

```bash
python api.py
```
_The API will be available at http://localhost:8000._

---

## 🔄 Usage Workflow

| Step | Action | When to do it? |
| :--- | :--- | :--- |
| **1. Scraping** | `python utils/dsa_scraper.py <URL>` | When you find a new set of problems the AI should learn. |
| **2. Preprocessing** | `python utils/dsa_preprocess.py` | After scraping, to prepare data for fine-tuning. |
| **3. Training** | `python training/train.py` | To bake that new knowledge into the AI's "brain." |
| **4. Inference** | `python api.py` | **Daily use**: To start the server and ask questions. |

---

## 🛠 Usage Examples

#### Get a DSA Problem Solution
```bash
curl.exe -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"How to implement a Threaded Binary Tree?\"}"
```

#### Train on new DSA Content
```bash
curl.exe -X POST "http://localhost:8000/train?url=https://www.geeksforgeeks.org/dynamic-programming/"
```

---

## 📡 API Reference

The Dsa LLM exposes a REST API built with FastAPI. By default, it runs on **port 8000**.

### 1. **Chat Endpoint** (`POST /chat`)
Generates a DSA solution and complexity analysis for a given prompt.

**Request Body:**
```json
{
  "prompt": "String: The coding problem or question",
  "max_tokens": "Integer: Limits response length (Default: 768)",
  "temperature": "Float: Controls creativity. Lower (0.1-0.5) is better for logic. (Default: 0.5)"
}
```

**Sample Response:**
```json
{
  "response": "To implement a LRU Cache, we use a Hash Map and a Doubly Linked List...\n\n```python\nclass LRUCache:\n    ...\n```\n\n**Complexity:**\n- Time: $O(1)$\n- Space: $O(N)$",
  "time_taken": 12.45
}
```

### 2. **Training Endpoint** (`POST /train`)
Triggers a background pipeline to scrape a URL, preprocess the data, and fine-tune the model.

**Query Parameters:**
- `url`: The web address to learn from.

**Sample Response:**
```json
{
  "message": "DSA training started",
  "url": "https://example.com/dsa-article"
}
```

### 3. **Interactive Documentation**
Once the server is running, you can access the full interactive API docs (Swagger UI) at:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker Deployment

To run the Dsa LLM in a containerized environment:

```bash
docker compose up --build
```

Credits: Gaurav Patel

