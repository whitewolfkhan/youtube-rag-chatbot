# 🎥 YouTube RAG Chatbot

An AI-powered chatbot that lets you ask questions about any YouTube video! Simply paste a YouTube URL, and the chatbot will extract the transcript, create embeddings, and answer your questions using Retrieval-Augmented Generation (RAG).

---

## ✨ Features

- **Ask Questions About Any Video** - Paste a YouTube URL and start asking questions
- **Context-Aware Answers** - Uses RAG to provide accurate answers based on video content
- **Source Timestamps** - Get timestamps linking to specific parts of the video
- **Chat-Style Interface** - Easy-to-use conversational interface
- **Multi-Language Support** - Works with videos in multiple languages
- **Fast & Efficient** - Powered by Groq's lightning-fast LLM inference

---

## 🎬 How It Works

```
User enters YouTube URL
        ↓
Transcript is extracted
        ↓
Text is split into chunks
        ↓
Embeddings are created
        ↓
Stored in FAISS vector database
        ↓
Semantic search retrieves relevant chunks
        ↓
LLM generates contextual answers
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **LLM** | Groq (Llama 3.1) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Database** | FAISS |
| **Transcript Extraction** | YouTube Transcript API |
| **Framework** | LangChain |

---

## Screenshots
<img width="1366" height="731" alt="Screenshot (291)" src="https://github.com/user-attachments/assets/bb1e77a7-eb9b-4744-93eb-397a832f50c2" />
<img width="1366" height="728" alt="Screenshot (292)" src="https://github.com/user-attachments/assets/54828321-5274-4ad2-a372-9601af01bb7e" />
<img width="1366" height="728" alt="Screenshot (293)" src="https://github.com/user-attachments/assets/fd78f1ed-7fd7-48e6-bd54-8a764d642b32" />



## 📋 Prerequisites

- Python 3.9 or higher
- Groq API Key (free at [console.groq.com](https://console.groq.com))

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Get your free Groq API Key:**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Copy and paste into your `.env` file

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 💡 Usage

1. **Open the app** in your browser
2. **Paste a YouTube URL** in the input field
3. **Click "Load Video"** to process the transcript
4. **Ask questions** about the video content
5. **Get answers** with source timestamps

### Example Questions

- "What is this video about?"
- "What are the key points discussed?"
- "Explain the main concepts covered"
- "What is [specific topic]?"

---

## 📁 Project Structure

```
youtube-rag-chatbot/
│
├── app.py              # Streamlit UI application
├── transcript.py       # YouTube transcript extraction
├── embeddings.py       # Embedding generation
├── vectorstore.py      # FAISS vector database operations
├── rag_chain.py        # RAG pipeline with Groq LLM
├── config.py           # Configuration settings
├── utils.py            # Helper functions
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not in git)
└── README.md           # This file
```

---

## ⚙️ Configuration

Edit `config.py` to customize the app:

```python
# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Embedding model
LLM_MODEL = "llama-3.1-8b-instant"      # Groq LLM model
LLM_TEMPERATURE = 0.7                    # Response creativity (0-1)
LLM_MAX_TOKENS = 2048                    # Max response length

# Chunking Settings
CHUNK_SIZE = 1000                        # Characters per chunk
CHUNK_OVERLAP = 200                      # Overlap between chunks

# Retrieval Settings
TOP_K_RESULTS = 6                        # Number of chunks to retrieve
```

---


## ⚠️ Known Limitations

- **Cloud Deployment**: YouTube may block requests from cloud servers (AWS, GCP, etc.)
- **Videos without Captions**: Works only with videos that have subtitles/captions
- **Private Videos**: Cannot access private or unlisted videos



---


## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---




