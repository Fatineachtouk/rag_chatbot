from langchain_ollama import OllamaLLM
import numpy as np
import re
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from datasets import Dataset
import pandas as pd

# Importing RAG functions
from ragg import (
    load_vectorstore,
    answer_generation,
    embedding_model,
    top_k
)

file_ids = ["002", "005", "004"]

test_cases = [
        {
            "question": "ما هو رقم البطاقة الوطنية؟",  
            "ground_truth": "K01234567",
            "metadata": {"file_id": "002"}
        },
        {
            "question": "Quel est le nom du propriétaire de la carte nationale?",
            "ground_truth": "MOUHCINE TEMSAMAN",
            "metadata": {"file_id": "002"}
        },
        {
            "question": "Quelle est la date de naissance?",
            "ground_truth": "29.11.1978",
            "metadata": {"file_id": "002"}
        },
        {
            "question": "ما هي المدينة؟",
            "ground_truth": "طنجة اصيلة",
            "metadata": {"file_id": "002"}
        },
        {
            "question": "Quelle est la valeure de TVA dans la facture de NOA ANDRIEUX?",
            "ground_truth": "1040€",
            "metadata": {"file_id": "005"}
        },
        {
            "question": "Quels sont les achats de NOA ANDRIEUX?",
            "ground_truth": "Identitévisuelle, Formation des équipes",
            "metadata": {"file_id": "005"}
        },
        {
            "question": "Quelle est la date d'echeance ddans la facture de NOA ANDRIEUX?",# des erreurs de frappe expres 
            "ground_truth":"30/11/2035",
            "metadata": {"file_id": "005"}
        }
]



def rag_outputs(test_cases, accessible_file_ids):
    """
    Run the RAG pipeline and collect all outputs needed for evaluation
    """
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    vectorstore = load_vectorstore(embedding_model)
    
    for test_case in test_cases:
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        answer = answer_generation(question, accessible_file_ids)
        
        # Extract contexts
        docs = vectorstore.similarity_search(
            question, 
            k=top_k, 
            filter={"file_id": {"$in": accessible_file_ids}}
        )
        contexts = [doc.page_content for doc in docs]
        
        # Store for evaluation
        evaluation_data["question"].append(question)
        evaluation_data["answer"].append(answer)
        evaluation_data["contexts"].append(contexts)  # List of retrieved chunks
        evaluation_data["ground_truth"].append(ground_truth)
    
    return evaluation_data  



def clean_answer(ans):
    """
    Remove extra polite phrases, trailing explanations, or redundant spaces.
    """
    ans = re.sub(r"(Let me know.*$)", "", ans, flags=re.IGNORECASE).strip()
    ans = re.sub(r"\s+", " ", ans)  # normalize spaces
    return ans


def ragas(accessible_file_ids, model_name="gemma2"):
    # Run RAG pipeline to get generated answers, contexts, and ground truth
    evaluation_data = rag_outputs(test_cases, accessible_file_ids)

    # Clean the answers before evaluation
    evaluation_data["answer"] = [clean_answer(a) for a in evaluation_data["answer"]]

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(evaluation_data)

    # Load evaluation model
    llm = OllamaLLM(model=model_name)

    # Evaluate
    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision, answer_correctness],
        llm=llm,
        embeddings=embedding_model
    )

    # Convert results to DataFrame for detailed view
    df = result.to_pandas()

    print("\n" + "="*60)
    print("RAGAS EVALUATION SCORES")
    print("="*60)
    print("\nDetailed Results:")
    print(df)

    # Aggregate scores
    print("\nAggregate Scores:")
    print(f"Answer Correctness: {np.nanmean(result['answer_correctness']):.3f}")
    print(f"Faithfulness: {np.nanmean(result['faithfulness']):.3f}")
    print(f"Answer Relevancy: {np.nanmean(result['answer_relevancy']):.3f}")
    print(f"Context Recall: {np.nanmean(result['context_recall']):.3f}")
    print(f"Context Precision: {np.nanmean(result['context_precision']):.3f}")

    return result


if __name__ == "__main__":
    ragas(file_ids, model_name="gemma2")