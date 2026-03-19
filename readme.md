# 🤖 RAG PDF Chatbot

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload a PDF and ask questions based on its content.

This project uses **LangChain, HuggingFace embeddings, and FAISS vector database** to retrieve relevant information and generate accurate answers.

---

## 🚀 Features

* 📄 Upload any PDF document
* 🔍 Extract and process text using chunking
* 🧠 Convert text into embeddings
* 📚 Store embeddings in FAISS vector database
* 💬 Ask questions and get context-aware answers
* ⚡ Fast and interactive UI using Streamlit

---

## 🏗️ Tech Stack

* **Python**
* **Streamlit** – UI
* **LangChain** – RAG pipeline
* **HuggingFace** – Embeddings / LLM
* **FAISS** – Vector database
* **PDFPlumber** – PDF text extraction

---

## 📂 Project Structure

```
chatbot/
│── app.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
```

Activate:

* Windows:

```bash
.venv\Scripts\activate
```

* Mac/Linux:

```bash
source .venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set Environment Variables

Create a `.env` file and add:

```
HUGGINGFACEHUB_API_TOKEN=your_api_key
```

---

### 5️⃣ Run the application

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app can be easily deployed using **Streamlit Cloud**:

1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Select your repo and deploy
4. Add your API key in **Secrets**

---

## 🧠 How It Works (RAG Pipeline)

1. **PDF → Text Extraction** using PDFPlumber
2. **Text → Chunks** using RecursiveCharacterTextSplitter
3. **Chunks → Embeddings** using HuggingFace
4. **Store in FAISS** vector database
5. **User Query → Similarity Search**
6. **Relevant Context + Query → LLM → Answer**

---

## 📌 Future Improvements

* ✅ Chat history memory
* ✅ Multi-PDF support
* ✅ Better UI/UX
* ✅ Add streaming responses
* ✅ Deploy with FastAPI backend

---

## 👨‍💻 Author

Gokul R
Gen AI Developer

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
