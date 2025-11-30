# Company Knowledge Base Agent 🤖

## Overview
This project is an AI-powered Knowledge Base Agent that answers questions about company policies, leave rules, work-from-home, benefits, and office timings.  
It uses a small internal knowledge base (`company_policies.txt`) and retrieves relevant information to generate accurate responses.

## Features
- Ask natural language questions about policies.
- Retrieves relevant policy sections using embeddings and vector search.
- Generates clear answers using OpenAI GPT.
- Simple Streamlit web UI.
- Runs locally or on Streamlit Cloud.

## Tech Stack
- **Language:** Python
- **Model:** OpenAI GPT (gpt-4o-mini)
- **Frameworks:** Streamlit, LangChain
- **Vector DB:** ChromaDB
- **Documents:** Plain text file (`data/company_policies.txt`)

## How It Works (Architecture)
1. Documents are loaded and split into chunks.
2. Chunks are converted into embeddings and stored in ChromaDB.
3. When the user asks a question, the most relevant chunks are retrieved.
4. The question + retrieved context are passed to the OpenAI GPT model.
5. The model returns a final answer that is displayed in the UI.

## Setup & Run Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/knowledge-base-agent.git
cd knowledge-base-agent
