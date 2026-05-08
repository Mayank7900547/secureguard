import pypdf
import os
import json

downloads_path = r"C:\Users\Mayank\Downloads"
pdf_files = [
    "1-s2.0-S187705092030065X-main.pdf",
    "Real-time-Credit-Card-Fraud-Detection-Using-Machine-Learning.pdf",
    "Credit Card Fraud Detection with Automated Machine Learning Systems.pdf",
    "s44230-022-00004-0.pdf",
    "electronics-14-01754.pdf",
    "journal.pone.0326975.pdf",
    "computers-14-00437.pdf",
    "s40537-022-00573-8.pdf",
    "Credit_Card_Fraud_Detection_Using_State-of-the-Art_Machine_Learning_and_Deep_Learning_Algorithms.pdf"
]

results = {}

for filename in pdf_files:
    file_path = os.path.join(downloads_path, filename)
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                text = ""
                # Get text from first 2 pages
                for i in range(min(2, len(reader.pages))):
                    text += reader.pages[i].extract_text() + "\n"
                results[filename] = text[:3000] # Limit to 3000 chars per paper
        else:
            results[filename] = "FILE NOT FOUND"
    except Exception as e:
        results[filename] = f"ERROR: {str(e)}"

with open("pdf_analysis.json", "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4)
