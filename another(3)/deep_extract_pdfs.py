import pypdf
import os
import json
import re

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

def find_info(text, page_num):
    # Looking for Accuracy, Precision, Recall
    acc = re.search(r'(Accuracy|Acc)[:\s]+(\d+\.\d+)', text, re.I)
    pre = re.search(r'(Precision|Pre)[:\s]+(\d+\.\d+)', text, re.I)
    rec = re.search(r'(Recall|Rec)[:\s]+(\d+\.\d+)', text, re.I)
    
    # Looking for headings
    headings = re.findall(r'^(\d+\.\s+[A-Z\s]+)', text, re.M)
    
    return {
        "page": page_num + 1,
        "acc": acc.group(2) if acc else "N/A",
        "pre": pre.group(2) if pre else "N/A",
        "rec": rec.group(2) if rec else "N/A",
        "headings": headings
    }

results = {}

for filename in pdf_files:
    file_path = os.path.join(downloads_path, filename)
    results[filename] = {"metrics": [], "summary": ""}
    try:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            # Sample first 2 and last 3 pages
            target_pages = list(range(min(2, len(reader.pages)))) + list(range(max(0, len(reader.pages)-3), len(reader.pages)))
            target_pages = list(set(target_pages))
            
            for p in target_pages:
                text = reader.pages[p].extract_text()
                info = find_info(text, p)
                results[filename]["metrics"].append(info)
                if p < 2:
                    results[filename]["summary"] += text[:1000]
    except Exception as e:
        results[filename] = f"ERROR: {str(e)}"

with open("deep_pdf_analysis.json", "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4)
