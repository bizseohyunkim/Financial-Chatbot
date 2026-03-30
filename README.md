## Live Demo
[Financial Document Chatbot](https://financial-chatbot-6y6zqkb5ywk2eotn7x74zv.streamlit.app)

---

# Financial Document Intelligence Chatbot

AI-powered document QA system for financial product analysis.  
금융 문서를 업로드하면 RAG 기술을 활용하여 질문에 자동으로 답변하는 챗봇입니다.

---

## Overview

This system allows users to upload financial product documents (PDF) and ask questions in natural language. The chatbot retrieves relevant context from the document and generates accurate answers using Google Gemini.

금융 상품 설명서, 약관, 보고서 등 PDF 문서를 업로드하고 자연어로 질문하면 문서 내용을 기반으로 정확한 답변을 제공합니다.

---

## Architecture
```
PDF Upload → Text Chunking → Vector Embedding → ChromaDB
                                                      ↓
User Query → Embedding → Similarity Search → Gemini 1.5 Flash → Response
```

---

## Features

- PDF document upload and automatic parsing
- Semantic search using vector embeddings
- Context-aware answers powered by Gemini 1.5 Flash
- Clean, professional UI built with Streamlit
- Supports Korean and English queries

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_1.5-4285F4?style=flat&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat&logoColor=white)

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/bizseohyunkim/Financial-Chatbot.git
cd Financial-Chatbot
```

**2. Install dependencies**
```bash
pip install langchain langchain-google-genai chromadb pypdf streamlit python-dotenv
```

**3. Set up API key**
```bash
# Create .env file
GOOGLE_API_KEY=your_gemini_api_key_here
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## Usage

1. Upload a financial PDF document from the sidebar
2. Wait for the document to be processed
3. Type your question in the chat input
4. Receive context-aware answers based on the document

---

## Note

- API key is required (Google AI Studio — free tier available)
- `.env` file is excluded from version control
- Responses are AI-generated and should be verified against official documents
