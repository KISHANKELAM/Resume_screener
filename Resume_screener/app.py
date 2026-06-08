import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import json
import os
import fitz  # PyMuPDF for PDFs
import docx
from PIL import Image
import pytesseract
import io

# ------------------------
# Load Artifacts
# ------------------------
SAVE_DIR = "./artifacts"

tokenizer = BertTokenizer.from_pretrained(SAVE_DIR)
with open(os.path.join(SAVE_DIR, "config.json")) as f:
    artifacts = json.load(f)
label_map = artifacts["label_map"]
MAX_LEN = artifacts["MAX_LEN"]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = torch.softmax(self.attn(x), dim=1)
        return (x * scores).sum(dim=1)

# Model definition
class BERTBiLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.attn = Attention(512)
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        return self.attn(lstm_out)

    def forward(self, r_input_ids, r_mask, j_input_ids, j_mask):
        r_vec = self.encode(r_input_ids, r_mask)
        j_vec = self.encode(j_input_ids, j_mask)
        return self.fc(torch.cat((r_vec, j_vec), dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTBiLSTMAttention()
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model.pth"), map_location=device))
model.to(device)
model.eval()

# ------------------------
# Utility Functions
# ------------------------
def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    
    if ext == ".pdf":
        text = ""
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
        return text
    
    elif ext in [".docx", ".doc"]:
        doc = docx.Document(io.BytesIO(file.read()))
        return " ".join([p.text for p in doc.paragraphs])
    
    elif ext in [".txt"]:
        return file.read().decode("utf-8")
    
    elif ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(file)
        return pytesseract.image_to_string(img)
    
    else:
        return ""

def predict_resume_fit(resume_text, job_text):
    resume_tokens = tokenizer(resume_text, padding='max_length', truncation=True,
                              max_length=MAX_LEN, return_tensors="pt")
    job_tokens = tokenizer(job_text, padding='max_length', truncation=True,
                           max_length=MAX_LEN, return_tensors="pt")

    r_ids, r_mask = resume_tokens["input_ids"].to(device), resume_tokens["attention_mask"].to(device)
    j_ids, j_mask = job_tokens["input_ids"].to(device), job_tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(r_ids, r_mask, j_ids, j_mask)
        pred = torch.argmax(outputs, dim=1).item()

    inv_label_map = {v: k for k, v in label_map.items()}
    return inv_label_map[pred]

# ------------------------
# Streamlit UI
# ------------------------
st.title("📄 AI-Powered Resume Screening")
st.write("Upload a **Resume** and a **Job Description** (PDF/DOCX/TXT/Image) OR paste text directly to check the fit.")

# --- Resume Input ---
st.subheader("Resume Input")
resume_file = st.file_uploader("Upload Resume File", type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg"])
resume_text_area = st.text_area("Or paste Resume text here")

# --- Job Description Input ---
st.subheader("Job Description Input")
jd_file = st.file_uploader("Upload Job Description File", type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg"])
jd_text_area = st.text_area("Or paste Job Description text here")

# --- Prediction ---
if st.button("Predict Fit"):
    resume_text = ""
    jd_text = ""

    if resume_file:
        resume_text = extract_text(resume_file)
    elif resume_text_area.strip():
        resume_text = resume_text_area.strip()

    if jd_file:
        jd_text = extract_text(jd_file)
    elif jd_text_area.strip():
        jd_text = jd_text_area.strip()

    if resume_text == "" or jd_text == "":
        st.error("❌ Please provide both Resume and Job Description (either upload or paste text).")
    else:
        result = predict_resume_fit(resume_text, jd_text)
        st.success(f"✅ Prediction: **{result}**")
