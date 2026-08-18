# AI Study Assistant

An AI-powered study assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions from study material and help students learn from their own documents.

## 📌 Project Overview

The AI Study Assistant allows users to interact with study material using natural language.

Instead of asking an AI model to answer only from its general knowledge, this project first retrieves relevant information from the provided study material and then uses that information to generate an answer.

The project currently uses a **Python Basics study material PDF** as the knowledge source.

The main RAG pipeline is:

User Question → Document Retrieval → Relevant Context → AI Model → Answer

---

## ✨ Features

- 📚 Ask questions about study material
- 🔎 Retrieve relevant information from the uploaded/available document
- 🤖 Generate AI-based answers using retrieved context
- 🧠 Uses Retrieval-Augmented Generation (RAG)
- 💾 Stores document embeddings using ChromaDB
- 📄 Supports PDF-based study material
- 🌐 Simple web interface using Flask
- 📝 Quiz generation functionality
- 🔐 API credentials are stored using environment variables

---

## 🛠️ Tech Stack

### Backend

- Python
- Flask

### AI / RAG

- Large Language Model (LLM)
- Embeddings
- Retrieval-Augmented Generation (RAG)

### Vector Database

- ChromaDB

### Document Processing

- PDF document processing
- Text chunking
- Embeddings

### Frontend

- HTML
- CSS
- JavaScript

### Development Tools

- Git
- GitHub
- Python Virtual Environment

---

## 🧠 How RAG Works in This Project

RAG stands for **Retrieval-Augmented Generation**.

It combines two important processes:

1. Retrieval
2. Generation

### Step 1: Load the study material

The study material is provided as a PDF.

Example:

`Python_Basics_Study_Material.pdf`

The application extracts the text from the document.

### Step 2: Split the document into chunks

Large documents are divided into smaller pieces called **chunks**.

For example:

```text
Python is a high-level programming language...