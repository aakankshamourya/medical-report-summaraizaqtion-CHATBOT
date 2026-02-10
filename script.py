from pypdf import PdfReader
from transformers import pipeline


# -------------------------------------------------
# FUNCTION: Extract text from PDF
# -------------------------------------------------
def extract_text(pdf_path: str) -> str:
    """
    Reads a PDF file and returns extracted text.
    Safe to import from other modules.
    """
    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# -------------------------------------------------
# MAIN EXECUTION (only runs if script executed directly)
# -------------------------------------------------
if __name__ == "__main__":

    pdf_path = "s41597-022-01899-x.pdf"

    print("Extracting text...")
    text = extract_text(pdf_path)

    print("Loading summarizer model...")

    summarizer = pipeline(
        task="text-generation",   # works reliably with latest transformers
        model="google/flan-t5-base"
    )

    # Summarize first chunk only (demo)
    chunk = text[:1000]

    prompt = f"Summarize the following medical text:\n{chunk}"

    result = summarizer(prompt, max_length=200)

    print("\n===== SUMMARY =====\n")
    print(result[0]["generated_text"])
