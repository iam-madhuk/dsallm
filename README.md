# 🎓 Dsa LLM: Specialist Algorithm AI API

Dsa LLM is a custom-trained, small-scale Large Language Model (LLM) designed specifically for **DSA Problem Solving**. It leverages base Qwen models and is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) to provide expert-level support in Data Structures and Algorithms.

---

## 🚀 Key Features

- **Algorithm Expert**: Built to be a Technical Interviewer, providing optimal code, approach explanations, and complexity analysis.
- **Contextual Memory (RAG)**: Uses **ChromaDB** with a specific `dsa_content` collection to store and retrieve algorithmic patterns.
- **REST API First**: Designed to be integrated into any coding platform or interview tool.
- **Speed Optimized**: Optimized for CPU/MPS (Metal Performance Shaders) execution for fast local inference.
- **LaTeX Complexity Output**: Automatically formats Big O notation using LaTeX (e.g., $O(N \log N)$).
- **End-to-End Pipeline**: Includes tools for web scraping and preprocessing DSA articles.

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

## 🛠 Usage Examples

#### Get a DSA Problem Solution
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "How to implement a Threaded Binary Tree?"}'
```

#### Train on new DSA Content
```bash
curl -X POST "http://localhost:8000/train?url=https://www.geeksforgeeks.org/dynamic-programming/"
```


---

## 🐳 Docker Deployment

To run the Dsa LLM in a containerized environment:

```bash
docker compose up --build
```
