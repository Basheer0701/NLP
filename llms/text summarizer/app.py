
from transformers import pipeline
import gradio as gr
import fitz  # PyMuPDF

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Extract text from uploaded PDF
def extract_text_from_pdf(file_obj):
    text = ""
    with fitz.open(file_obj.name) as doc:
        for page in doc:
            text += page.get_text()
    return text


# Summarizer function
def summarize_input(text, file):
    if file is not None:
        text = extract_text_from_pdf(file)

    if not text or len(text.strip()) == 0:
        return "❌ No valid text found."

    chunk_size = 3000
    overlap = 200
    summaries = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(f"🔹 {result[0]['summary_text']}")

    return "\n\n".join(summaries)


# Gradio UI
iface = gr.Interface(
    fn=summarize_input,
    inputs=[
        gr.Textbox(lines=10, label="📝 Enter text", placeholder="Paste long text here..."),
        gr.File(label="📄 Upload PDF", file_types=[".pdf"])
    ],
    outputs=gr.Textbox(label="📌 Summary"),
    title="🧠 Text & PDF Summarizer",
    description="Paste text or upload a PDF. Get an automatic summary using Facebook's BART model."
)

iface.launch(share=True)
print("✅ Gradio app has been launched.")