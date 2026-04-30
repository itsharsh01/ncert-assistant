import os
import asyncio
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import AnswerAccuracy, ContextPrecision, Faithfulness

from rag_pipeline.main import get_answer_fast

load_dotenv()

# We will use Gemini-2.5-Flash for the evaluation LLM
# wrapped properly for the newest Ragas framework
google_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
ragas_llm = LangchainLLMWrapper(google_llm)

# Initialize the new Ragas metric scorers
answer_accuracy = AnswerAccuracy(llm=ragas_llm)
context_precision = ContextPrecision(llm=ragas_llm)
faithfulness = Faithfulness(llm=ragas_llm)

questions = [
    {
        "user_input": "What is matter?",
        "reference": "Matter is anything that occupies space and has mass."
    },
    {
        "user_input": "Explain the difference between homogeneous and heterogeneous mixtures in detail.",
        "reference": "Homogeneous mixtures have a uniform composition throughout, while heterogeneous mixtures contain physically distinct parts and have non-uniform compositions."
    },
    {
        "user_input": "What is the formula to calculate the mass percentage of a solution?",
        "reference": "Mass percentage of a solution = (Mass of solute / Mass of solution) x 100"
    },
    {
        "user_input": "Who won the FIFA World Cup in 2022 and what was the final score?",
        "reference": "I don't know."
    }
]

async def evaluate():
    results_list = []
    
    print("Starting Ragas Evaluation...")
    for idx, item in enumerate(questions, 1):
        q = item["user_input"]
        ref = item["reference"]
        
        print(f"\\nEvaluating Q{idx}: {q}")
        
        # Get response and docs from our pipeline
        try:
            response, docs = get_answer_fast(q, save_dir="saved_pipeline")
        except Exception as e:
            response = f"Error: {e}"
            docs = []
            
        # Extract context texts correctly
        contexts = []
        for d in docs:
            if isinstance(d, tuple):
                contexts.append(d[1].page_content)
            else:
                contexts.append(d.page_content)
                
        # If pipeline safely bailed out, ensure contexts is empty
        if "I don't know" in response:
            contexts = []
            
        # Calculate metrics using the latest async ascore() method
        
        try:
            acc_result = await answer_accuracy.ascore(
                user_input=q, 
                response=response, 
                reference=ref
            )
            acc_score = acc_result.value
        except Exception as e:
            print(f"  [!] Error calculating AnswerAccuracy: {e}")
            acc_score = 0.0
            
        try:
            cp_result = await context_precision.ascore(
                user_input=q,
                response=response,
                retrieved_contexts=contexts,
                reference=ref
            )
            cp_score = cp_result.value
        except Exception as e:
            print(f"  [!] Error calculating ContextPrecision: {e}")
            cp_score = 0.0
            
        try:
            faith_result = await faithfulness.ascore(
                user_input=q,
                response=response,
                retrieved_contexts=contexts
            )
            faith_score = faith_result.value
        except Exception as e:
            print(f"  [!] Error calculating Faithfulness: {e}")
            faith_score = 0.0
            
        print(f"  -> Answer Accuracy: {acc_score}")
        print(f"  -> Context Precision: {cp_score}")
        print(f"  -> Faithfulness: {faith_score}")
            
        results_list.append({
            "question": q,
            "response": response,
            "reference": ref,
            "retrieved_contexts_count": len(contexts),
            "metrics": {
                "answer_accuracy": acc_score,
                "context_precision": cp_score,
                "faithfulness": faith_score
            }
        })
        
    with open("ragas_results.json", "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4)
        
    print("\\nSuccess! All metrics saved to 'ragas_results.json'")

if __name__ == "__main__":
    asyncio.run(evaluate())
