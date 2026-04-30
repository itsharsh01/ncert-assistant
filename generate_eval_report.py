import os
import time
from dotenv import load_dotenv
from rag_pipeline.main import get_answer_fast

load_dotenv()

questions = [
    # 1. Short
    "What is matter?",
    # 2. Long
    "Explain the difference between homogeneous and heterogeneous mixtures in detail.",
    # 3. Example
    "Give some examples of colloidal solutions from everyday life.",
    # 4. Formula
    "What is the formula to calculate the mass percentage of a solution?",
    # 5. Definition
    "Define the Tyndall effect.",
    # 6. Process
    "How can we separate a mixture of two immiscible liquids?",
    # 7. Comparison
    "Differentiate between a physical change and a chemical change.",
    # 8. Reasoning
    "Why does our palm feel cold when we put some acetone on it?",
    # 9. Facts
    "What is the boiling point and melting point of water?",
    # 10. Property
    "What are the main properties of a suspension?",
    # 11. Concept
    "Explain evaporation and the factors affecting it.",
    # 12. Unrelated (Should trigger 'I don't know.')
    "Who won the FIFA World Cup in 2022 and what was the final score?"
]

print("Starting to generate evaluation report for 12 questions...")
print("Please wait, querying the Gemini API for each question...")

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>12-Question Evaluation Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f8fafc;
            --text-main: #1e293b;
            --card-bg: #ffffff;
            --accent: #2563eb;
            --border: #e2e8f0;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 40px 20px;
            margin: 0;
            line-height: 1.6;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 50px;
        }
        .qa-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .question-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }
        .question {
            font-size: 1.3rem;
            font-weight: 600;
            color: #0f172a;
            margin-top: 0;
            margin-bottom: 20px;
        }
        .answer {
            background: #f1f5f9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid var(--accent);
            font-size: 1.05rem;
            color: #334155;
            white-space: pre-wrap;
        }
        .idk-answer {
            border-left-color: #ef4444;
            background: #fef2f2;
        }
        .context-found {
            font-size: 0.85rem;
            color: #10b981;
            margin-top: 15px;
            font-weight: 500;
        }
        .no-context {
            font-size: 0.85rem;
            color: #ef4444;
            margin-top: 15px;
            font-weight: 500;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>NCERT Science Class 9 <br> RAG Evaluation Report</h1>
"""

for idx, q in enumerate(questions, 1):
    print(f"[{idx}/12] Processing Question: {q}")
    try:
        answer, docs = get_answer_fast(q, save_dir="saved_pipeline")
    except Exception as e:
        answer = f"Error occurred: {e}"
        docs = []
        
    is_idk = ("I don't know" in answer or len(docs) == 0)
    answer_class = "answer idk-answer" if is_idk else "answer"
    
    if len(docs) == 0:
        context_msg = "<div class='no-context'>⚠️ No valid context chunks found for this query. System safely defaulted to 'I don't know.'</div>"
    else:
        context_msg = f"<div class='context-found'>✓ Found {len(docs)} relevant textbook chunks to ground this answer.</div>"
    
    html_content += f"""
    <div class="qa-card">
        <div class="question-label">Question {idx}</div>
        <h2 class="question">{q}</h2>
        <div class="{answer_class}">{answer.replace('<', '&lt;').replace('>', '&gt;')}</div>
        {context_msg}
    </div>
    """
    # Sleep to avoid hitting free-tier Gemini rate limits (15 RPM)
    time.sleep(4)

html_content += """
</div>
</body>
</html>
"""

with open("12_qa_eval_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("\\nSuccess! The report has been generated as '12_qa_eval_report.html'")
