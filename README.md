# AI-Powered Resume Screener

An end-to-end artificial intelligence application designed to streamline the recruitment process. This tool automatically extracts text from uploaded resume documents and matches them against specific job descriptions using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) deep learning classification pipeline.

## 🚀 Features
* **Interactive Web Dashboard:** Built a clean, real-time user interface using **Streamlit** for seamless file uploading and visualization.
* **Semantic Match Analysis:** Implemented **PyTorch** and **HuggingFace Transformers** to evaluate context-aware similarity rather than simple keyword matching.
* **Text Preprocessing Pipeline:** Cleans messy resume data automatically, discarding formatting noise, special characters, and structural artifacts.

## 🛠️ Tech Stack
* **Language:** Python
* **Web Interface Framework:** Streamlit
* **Machine Learning & AI:** PyTorch, HuggingFace Transformers (BERT)

## 📦 Repository Optimization & Challenges Overcame
During development, the local repository tracking state accumulated heavy machine learning model weights (`artifacts/` directory) and compressed model backups, ballooning the project history cache to nearly 800 MB. 

To ensure clean project delivery and optimized cloud hosting, the repository was successfully refactored by:
1. Configuring strict tracking filters in a custom `.gitignore` to exclude large binary files and machine learning weight blocks.
2. Purging Git's tracking database cache via terminal optimization to reduce the final deployment bundle from **789 MiB down to just 49 KiB**.
3. Mirroring production-ready code structures directly onto the repository root for immediate scannability.

## 🔧 Installation & Local Setup

1. **Clone the repository:**
```bash
   git clone [https://github.com/KISHANKELAM/Resume_screener.git](https://github.com/KISHANKELAM/Resume_screener.git)
   cd Resume_screener
Set up a virtual environment:

Bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
Install dependencies:

Bash
   pip install -r requirements.txt
Run the application:

Bash
   streamlit run app.py
