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

---

# 🧠 System Workflow 

---

# 📥 Ingestion Pipeline (`ingest.py`)

The ingestion module prepares the knowledge base for retrieval.

### Workflow

```text
Knowledge Base Documents
        │
        ▼
Load Markdown Files
        │
        ▼
Chunk Documents
        │
        ▼
Generate Embeddings
        │
        ▼
Store in Vector Database
        │
        ▼
Ready for Retrieval
```

### Responsibilities

* Reads all knowledge-base documents
* Splits documents into chunks
* Creates vector embeddings
* Builds the retrieval index
* Stores vectors for semantic search

### Run

```bash
python ingest.py
```

---

# 🔎 Question Answering (`answer.py`)

This module performs retrieval and answer generation.

### Workflow

```text
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Search Vector Database
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Build Context
      │
      ▼
Send Context + Query to LLM
      │
      ▼
Generate Answer
```

### Responsibilities

* Accepts user questions
* Performs semantic retrieval
* Collects relevant context
* Sends context to the LLM
* Produces grounded responses

### Run

```bash
python answer.py
```

---

# 💬 Chat Interface (`app.py`)

Creates a gradio app showing the chat interface 

### Run

```bash
python app.py
```

# 📸 Application Screenshot

![App Screenshot](https://github.com/SujithVarma-ai/Advanced-RAG-Knowledge-Base/blob/main/Screenshot%202026-06-08%20160135.png)

## Evaluation Workflow

```text 
 ingest.py
     │
     ▼
 answer.py
     │
     ▼
  app.py
```

---

# 📊 Evaluation Pipeline

The project includes an automated evaluation framework for measuring RAG quality.

---

## Test Dataset (`tests.jsonl`)

Contains:

* Test Questions
* Ground Truth Answers
* Evaluation Samples

Used to benchmark the RAG system.

---

## Testing Module (`test.py`)

### Responsibilities

* Loads evaluation questions
* Sends queries to the RAG pipeline
* Collects generated answers
* Creates evaluation outputs

### Run

```bash
python test.py
```

---

## Evaluation Module (`eval.py`)

### Responsibilities

* Compares generated answers
* Measures relevance
* Measures accuracy
* Measures completeness

### Run

```bash
python eval.py
```

---

## Judge Evaluator (`evaluator.py`)

### Responsibilities

* Uses an LLM-as-a-Judge approach
* Evaluates answer quality
* Produces evaluation scores
* Generates performance reports

### Run

```bash
python evaluator.py
```

# 📸 Application Screenshot

![App Screenshot](https://github.com/SujithVarma-ai/Advanced-RAG-Knowledge-Base/blob/main/Screenshot%202026-06-08%20155908.png)
![App Screenshot](https://github.com/SujithVarma-ai/Advanced-RAG-Knowledge-Base/blob/main/Screenshot%202026-06-08%20155926.png)

---

## Evaluation Workflow

```text
tests.jsonl
     │
     ▼
  test.py
     │
     ▼
Generated Answers
     │
     ▼
   eval.py
     │
     ▼
evaluator.py
     │
     ▼
Evaluation Scores
```

# 🚀 Advanced RAG Features

This project extends traditional RAG by incorporating:

## 1. Query Rewriting

The original user query is rewritten into a more retrieval-friendly form.

### Example

**User Query**

```text
Tell me about Health11m
```

**Rewritten Query**

```text
Provide details about the Health11m insurance product, including its features, coverage, and target customers.
---


## 2. Reranking

After retrieval, documents are reranked according to relevance.

### Example

**Retrieved Documents**

```text
1. Product Overview
2. Employee Profile
3. Contract Information
4. Health11m Product Details
```

**After Reranking**

```text
1. Health11m Product Details
2. Product Overview
3. Contract Information
4. Employee Profile
---

# 🏗️ Final Advanced RAG Workflow

```text
User
 │
 ▼
Question
 │
 ▼
Query Rewriting
 │
 ▼
Retriever
 │
 ▼
Reranking
 │
 ▼
LLM
 │
 ▼
Answer
```


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
git clone https://github.com/SujithVarma-ai/Advanced-RAG-Knowledge-Base.git
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
