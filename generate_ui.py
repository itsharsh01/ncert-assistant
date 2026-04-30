import os
from dotenv import load_dotenv
from rag_pipeline.main import RAGPipeline

load_dotenv()

# Initialize the pipeline smartly
print("Initializing RAG Pipeline...")
pipeline = RAGPipeline()

# Check if we already built the models
if not os.path.exists("saved_pipeline"):
    print("Pre-trained models not found. Building from scratch... (This will take a few minutes)")
    pipeline.build_and_save(pdf_path="ncert-9-1-30.pdf", save_dir="saved_pipeline")
else:
    print("Pre-trained models found! Loading instantly...")
    pipeline.load_pipeline(save_dir="saved_pipeline")

my_query = "What is the difference between homogeneous and heterogeneous mixtures?"

print(f"Query: {my_query}")

print("Running Method 1...")
results_1 = pipeline.query_comparison_without_reranker(my_query)

print("Running Method 2...")
results_2 = pipeline.query_with_reranker(my_query)

print("Running Method 3...")
results_3 = pipeline.query_with_positive_scores_sorting(my_query)

# UI Generation
def create_tabs(results, section_id):
    if not results:
        return "<p style='color: #888;'>No results found.</p>"
        
    tabs_nav = []
    tabs_content = []
    
    for i, res in enumerate(results):
        if isinstance(res, tuple):
            score, doc = res
            title = f"Result {i+1} <span class='score'>Score: {score:.3f}</span>"
            content = doc.page_content
        else:
            doc = res
            title = f"Result {i+1}"
            content = doc.page_content
            
        active_class = "active" if i == 0 else ""
        
        # HTML escaping for content
        content_escaped = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        
        tabs_nav.append(f'<button class="tab-btn {active_class}" onclick="openTab(event, \'{section_id}-tab{i}\')">{title}</button>')
        tabs_content.append(f'<div id="{section_id}-tab{i}" class="tab-content {active_class}"><p>{content_escaped}</p></div>')
        
    nav_html = f'<div class="tab-nav">{"".join(tabs_nav)}</div>'
    content_html = f'<div class="tab-container">{"".join(tabs_content)}</div>'
    return nav_html + content_html

html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NCERT Assistant Results</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
    :root {{
        --bg-color: #0b1120;
        --card-bg: rgba(30, 41, 59, 0.4);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --accent: #6366f1;
        --accent-hover: #4f46e5;
        --tab-bg: rgba(255,255,255,0.03);
        --border: rgba(255,255,255,0.08);
    }}
    body {{
        font-family: 'Inter', sans-serif;
        background: var(--bg-color);
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.15), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.15), transparent 25%);
        color: var(--text-main);
        margin: 0;
        padding: 50px 20px;
        min-height: 100vh;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
    }}
    header {{
        text-align: center;
        margin-bottom: 50px;
    }}
    h1 {{
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }}
    .query-box {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 25px 30px;
        margin-bottom: 40px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }}
    .query-box::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: linear-gradient(to bottom, #818cf8, #f472b6);
    }}
    .query-box h3 {{ margin-top: 0; color: #cbd5e1; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 10px; }}
    .query-box p {{ font-size: 1.35rem; margin: 0; font-weight: 300; line-height: 1.5; color: #fff; }}
    
    .section {{
        background: var(--card-bg);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 40px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }}
    .section h2 {{
        margin-top: 0;
        font-size: 1.4rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 15px;
        margin-bottom: 25px;
        color: #e2e8f0;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .section h2 span.icon {{
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        width: 32px; height: 32px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 8px; font-size: 0.9rem; font-weight: 700;
    }}
    
    .tab-nav {{
        display: flex;
        gap: 12px;
        margin-bottom: 25px;
        overflow-x: auto;
        padding-bottom: 5px;
    }}
    .tab-btn {{
        background: var(--tab-bg);
        border: 1px solid var(--border);
        color: var(--text-muted);
        padding: 12px 20px;
        border-radius: 10px;
        cursor: pointer;
        font-family: inherit;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .tab-btn:hover {{
        background: rgba(255,255,255,0.08);
        color: var(--text-main);
        transform: translateY(-2px);
    }}
    .tab-btn.active {{
        background: var(--accent);
        color: white;
        border-color: var(--accent-hover);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }}
    .score {{
        background: rgba(0,0,0,0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    .tab-btn.active .score {{ background: rgba(0,0,0,0.2); }}
    
    .tab-content {{
        display: none;
        animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        line-height: 1.8;
        color: #cbd5e1;
        background: rgba(0,0,0,0.25);
        padding: 30px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.03);
        font-size: 1.05rem;
    }}
    .tab-content.active {{
        display: block;
    }}
    
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.1); }}
    ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.1); border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.2); }}
</style>
</head>
<body>

<div class="container">
    <header>
        <h1>Intelligence Results</h1>
        <p style="color: var(--text-muted); font-size: 1.1rem;">RAG Pipeline Retrieval Methodologies Comparison</p>
    </header>

    <div class="query-box">
        <h3>User Query</h3>
        <p>"{my_query}"</p>
    </div>

    <div class="section">
        <h2><span class="icon">1</span> Hybrid Comparison (BM25 + Vectors)</h2>
        {create_tabs(results_1, "sec1")}
    </div>

    <div class="section">
        <h2><span class="icon">2</span> Cross-Encoder Re-ranker</h2>
        {create_tabs(results_2, "sec2")}
    </div>

    <div class="section">
        <h2><span class="icon">3</span> Positive Scores Only (Filtered)</h2>
        {create_tabs(results_3, "sec3")}
    </div>
</div>

<script>
function openTab(evt, tabId) {{
    var section = document.getElementById(tabId).closest('.section');
    var tabContents = section.getElementsByClassName("tab-content");
    var tabBtns = section.getElementsByClassName("tab-btn");
    
    for (var i = 0; i < tabContents.length; i++) {{
        tabContents[i].classList.remove("active");
        tabBtns[i].classList.remove("active");
    }}
    
    document.getElementById(tabId).classList.add("active");
    evt.currentTarget.classList.add("active");
}}
</script>

</body>
</html>
"""

with open("results.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("\nSuccess! Generated 'results.html'. Open this file in your browser to view the beautiful UI.")
