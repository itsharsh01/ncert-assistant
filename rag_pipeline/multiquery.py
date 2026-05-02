import os
import google.generativeai as genai

def generate_multi_queries(query: str, n=3):
    """
    Uses the LLM to generate alternative phrasing of the query for better retrieval.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return [query]
        
    genai.configure(api_key=api_key)
    
    prompt = f"""You are an AI language model assistant. Your task is to generate {n} different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. DO NOT number them.

Original Question: {query}"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        # Parse the output
        queries = response.text.strip().split('\n')
        # Clean up any bullet points or numbering
        clean_queries = []
        for q in queries:
            cleaned = q.strip()
            if cleaned.startswith('- '): cleaned = cleaned[2:]
            if cleaned.startswith('* '): cleaned = cleaned[2:]
            if len(cleaned) > 5:
                clean_queries.append(cleaned)
                
        if not clean_queries:
            return [query]
            
        return clean_queries[:n]
    except Exception as e:
        print(f"MultiQuery Error: {e}")
        return [query]
