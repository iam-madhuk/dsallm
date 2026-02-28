# 🎓 Edufya & CodeEdu LLM: Private Educational AI APIs

EdufyaLLM is a suite of custom-trained, small-scale Large Language Models (LLMs) designed specifically for **Educational Tutoring**. It leverages base Qwen models and is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) to provide specialized support in Mathematics and Computer Science (DSA).

---

## 🚀 Key Features

- **Dual Specialist Model**: Includes a **Mathematics Tutor** and a **DSA Interview Specialist**.
- **REST API Architecture**: Two independent high-performance APIs for specialized tasks.
- **Contextual Memory (RAG)**: Uses **ChromaDB** with isolated collections (`educational_content` and `dsa_content`) to prevent knowledge leakage.
- **LaTeX Math Output**: Generates clean mathematical notation (e.g., $x^2 + y^2 = z^2$).
- **Algorithm Expert**: CodeEduLLM provides optimal code, approach explanations, and Time/Space complexity analysis.
- **Speed Optimized**: Optimized for CPU/MPS (Metal Performance Shaders) execution.
- **End-to-End Pipeline**: Includes tools for web scraping, data preprocessing, and model fine-tuning.

---

## 🛠 Project Structure

| Specialist | App File | Port | Goal |
| :--- | :--- | :--- | :--- |
| **Math Tutor** | `api.py` | `8000` | Step-by-step math proofs & logic. |
| **DSA Specialist** | `dsa_api.py` | `8001` | Optimal algorithms & complexity analysis. |

---

## 🚦 Getting Started

### 1. Installation

Clone the repo and install the requirements (optimized for CPU/Mac):

```bash
pip install -r requirements.txt
```

### 2. Running the APIs

#### Start the Math Tutor:
```bash
python api.py
```

#### Start the DSA Specialist:
```bash
python dsa_api.py
```

### 3. Usage Examples

#### Get a DSA Problem Solution (Port 8001)
```bash
curl -X POST "http://localhost:8001/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "How to implement a LRU Cache?"}'
```

#### Get Math Help (Port 8000)
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain the Pythagorean Theorem"}'
```

---

## 🐳 Docker Deployment

To run the APIs in a containerized environment (defaults to Math API):

```bash
docker compose up --build
```
