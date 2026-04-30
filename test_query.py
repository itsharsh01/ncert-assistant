from rag_pipeline.main import get_answer_fast
from dotenv import load_dotenv

load_dotenv()

# Instantly load models, search textbook, and generate Gemini response!
query = "What is the difference between homogeneous and heterogeneous mixtures?"
final_answer, used_documents = get_answer_fast(query)

print("=== FINAL AI ANSWER ===")
print(final_answer)
