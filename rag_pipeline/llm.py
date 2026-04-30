import os
import google.generativeai as genai

def generate_guardrailed_answer(query: str, retrieved_docs) -> str:
    """
    Generates a guardrailed answer using the Gemini API based strictly on the provided documents.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found in environment."
        
    genai.configure(api_key=api_key)
    
    # Extract text from retrieved documents
    # Note: retrieved_docs might be tuples of (score, doc)
    context_parts = []
    for item in retrieved_docs:
        if isinstance(item, tuple):
            context_parts.append(item[1].page_content)
        else:
            context_parts.append(item.page_content)
            
    context_text = "\n\n---\n\n".join(context_parts)
    
    if not context_text.strip():
         return "I could not find relevant information in the textbook to answer your question."

    prompt = f"""You are a strict, helpful educational assistant specifically built to answer student questions based on the NCERT textbook.
You must answer the user's question using ONLY the provided textbook context below.

CRITICAL RULES:
1. If the answer cannot be fully determined from the provided context, you MUST reply exactly with: "I'm sorry, but I cannot answer this based on the provided NCERT textbook."
2. Do not hallucinate or use any outside knowledge.
3. Keep the answer clear, structured, and easy for a student to understand.
4. If applicable, cite the relevant facts directly from the context.

=== TEXTBOOK CONTEXT ===
{context_text}
========================

User Question: {query}

Guardrailed Answer:"""

    try:
        # Use gemini-2.5-flash for fast and accurate responses
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response from Gemini: {e}"
