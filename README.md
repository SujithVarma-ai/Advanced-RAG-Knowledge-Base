# 🚀 Advanced RAG Knowledge Base

A Retrieval-Augmented Generation (RAG) system that combines structured knowledge retrieval with Large Language Models to answer questions accurately using company, employee, contract, and product information.

---

## 📌 Overview

This project demonstrates an advanced RAG pipeline that:

- Builds a knowledge base from multiple document collections
- Retrieves relevant information using semantic search
- Generates context-aware responses with an LLM
- Evaluates answer quality automatically
- Supports enterprise-style datasets

The knowledge base contains information about:

- Company Profiles
- Employees
- Contracts
- Products

---

## 📂 Project Structure

```text
Advanced-RAG-Knowledge-Base/
│
├── knowledge-base/
│   ├── company/
│   ├── contracts/
│   ├── employees/
│   └── products/
│
├── app.py
├── ingest.py
├── answer.py
├── eval.py
├── evaluator.py
├── ragadvanced.py
├── tests.jsonl
├── README.md
```

---

## 🧠 How the System Works

### Step 1: Knowledge Base Creation

Documents are organized into categories:

- Company Information
- Employee Profiles
- Business Contracts
- Product Details

Each document is stored as a Markdown file.

---

### Step 2: Ingestion

The ingestion pipeline:

- Reads documents
- Splits content into chunks
- Generates embeddings
- Stores vectors for retrieval

Run:

```bash
python ingest.py
```

---

### Step 3: Retrieval

When a user asks a question:

1. Query embedding is generated
2. Relevant documents are retrieved
3. Retrieved context is sent to the LLM

---

### Step 4: Answer Generation

The language model generates a response grounded in the retrieved knowledge.

Run:

```bash
python answer.py
```

---

### Step 5: Evaluation

Generated responses are automatically evaluated for:

- Accuracy
- Relevance
- Completeness
- Groundedness

Run:

```bash
python eval.py
```

or

```bash
python evaluator.py
```

---

## 📊 Knowledge Base Categories

### 🏢 Company

Contains:

- Company overview
- Culture
- Career information
- Business details

### 👨‍💼 Employees

Contains:

- Employee profiles
- Roles and responsibilities
- Organizational information

### 📑 Contracts

Contains:

- Business agreements
- Partnership contracts
- Service agreements

### 📦 Products

Contains:

- Product descriptions
- Features
- Business offerings

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Advanced-RAG-Knowledge-Base.git
```

Move into the project:

```bash
cd Advanced-RAG-Knowledge-Base
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Build the Vector Database

```bash
python ingest.py
```

### Ask Questions

```bash
python answer.py
```

### Run Evaluation

```bash
python eval.py
```

---

## 🧪 Testing

The repository includes:

```text
tests.jsonl
```

Used for evaluating system performance against predefined test cases.

---

## 🎯 Key Features

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector-Based Retrieval
- Enterprise Knowledge Base
- Automated Evaluation Pipeline
- Modular Architecture
- Scalable Document Management

---

## 🛠️ Technologies Used

- Python
- Large Language Models (LLMs)
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)
- JSONL Testing Framework
- Markdown Knowledge Base

---

## 📈 Future Improvements

- Web-based chat interface
- Advanced reranking
- Hybrid search
- Metadata filtering
- Multi-document reasoning
- Real-time knowledge updates

---

## 👨‍💻 Author

**Nadimpalli Raja Sujith Varma**

Advanced RAG Knowledge Base Project
