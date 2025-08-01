# RAG-Based Quiz Generator

**AI-assisted app to upload, process, and quiz from your own documents using ChromaDB and Llama 3.2.**

## 🚦 Features

- **Document Upload**: Accepts PDF, DOCX, PPTX, XLSX.
- **Selective Processing**: Pick specific pages, paragraphs, slides, or sheets.
- **Vector Store**: Text is chunked and embedded via MiniLM into ChromaDB.
- **Quiz Generation**: Uses Llama 3.2 (local) to generate multiple-choice questions (MCQ).
- **Export**: Download quizzes as plaintext.
- **Flexible UI**: Powered by Streamlit, runs in your browser.

## 📁 Files Overview

| File             | Description                                                                                                         |
|------------------|---------------------------------------------------------------------------------------------------------------------|
| `app.py`         | Main Streamlit app. Handles UI, file processing, text embedding, quiz generation, and quiz output.                   |
| `config.py`      | Configuration settings for host/port, model and embedding paths, ChromaDB, chunking, and quiz parameters. Ensures directories exist on import. |
| `run_quizgen.sh` | Helper script to start the app with Streamlit, logs to `./logs/`.                                                   |

## ⚡ Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- `chromadb`, `sentence-transformers`, `transformers`
- `PyPDF2`, `python-docx`, `openpyxl`, `python-pptx`, `pandas`
- PyTorch (GPU recommended for Llama 3.2)
- **Llama 3.2-3B-Instruct** model (download and update path in `config.py`)

### Install

pip install streamlit chromadb sentence-transformers transformers PyPDF2 python-docx openpyxl python-pptx pandas torch


## 🚀 Setup

1. **Clone the Repository**

    ```
    git clone <your-repo-url>
    cd <your-repo-dir>
    ```

2. **Model Files**

    Download [Llama 3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) locally and set `MODEL_NAME` in `config.py` to the correct path.

3. **Prep Directories**

    ```
    mkdir -p chroma_db logs
    ```

## 🏃 Running

### A. With Script

bash run_quizgen.sh

- The app starts in the background.
- Default URL: `http://<your-server-ip>:13000`

### B. Manual

streamlit run app.py --server.address 0.0.0.0 --server.port 13000


## 📝 Usage

1. Open `http://localhost:13000` (or replace `localhost` with your server IP).
2. Upload PDF, DOCX, PPTX, or XLSX files.
3. Optionally, specify which pages, slides, paragraphs, or sheets to process.
4. Click **Process Documents**.
5. Configure the quiz (number of questions, quiz title).
6. Click **Generate Quiz**.
7. Copy or download your MCQ quiz.
8. Use the sidebar to clear the database or generated questions as needed.

## 🛠️ Troubleshooting

- **Model fails to load**? Check the `MODEL_NAME` path in `config.py` and ensure there is enough GPU RAM.
- **Database issues**? Check write permissions for the `chroma_db` directory.
- **File problems**? Make sure documents are not corrupted and your input selection is valid.

## ✨ Credits

Developed by [Your Name/Team].

- Built with open-source: Streamlit, HuggingFace Transformers, ChromaDB, MiniLM, PyPDF2, python-docx, openpyxl, python-pptx, pandas.

## 📄 License

[Specify your license here.]

**Happy Quizzing!**

> _Adjust any file paths, ports, and instructions as needed for your deployment._
