import fitz
import os

def extract_text(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"OK: {os.path.basename(pdf_path)}")
    print(f"   Date: {metadata.get('creationDate', 'non disponible')}")
    print(f"   Pages: {doc.page_count}")

def process_folder(folder):
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder, filename)
            txt_path = pdf_path.replace(".pdf", ".txt")
            if not os.path.exists(txt_path):
                extract_text(pdf_path, txt_path)
            else:
                print(f"Skipped (already exists): {filename}")

print("=== BROCHURES ===")
process_folder("data/used/brochures")

print("\n=== FLYERS ===")
process_folder("data/used/flyers")