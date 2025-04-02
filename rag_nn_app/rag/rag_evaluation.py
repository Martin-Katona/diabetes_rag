import json
import logging
from rag_01 import query_rag_nested
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


# Suppress INFO messages from the 'httpx' logger
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer only with 'true' or 'false') Does the actual response match or is similar to expected response ? 
"""


def load_questions(json_file: str) -> list:
    """
    Load questions and expected responses from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        list: List of dictionaries containing questions and expected responses.
    """
    try:
        with open(json_file, "r") as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions from {json_file}.")
        return questions
    except Exception as e:
        logger.error(f"Failed to load questions from {json_file}: {e}")
        return []



def evaluate_response(expected_response: str, actual_response: str, eval_models=['mistral']) -> bool:
    """
    Evaluate if the actual response matches the expected response using one or three LLMs.
    If three LLMs are provided, a voting mechanism is used to determine the result.

    Args:
        expected_response (str): The expected response.
        actual_response (str): The actual response from the RAG model.
        eval_models (list): List of one or three LLM models to use for evaluation.

    Returns:
        bool: True if the response matches the expected response, False otherwise.
              If three models are provided, the result is based on majority voting.
    """
    try:
        # Format the evaluation prompt
        prompt = EVAL_PROMPT.format(
            expected_response=expected_response, actual_response=actual_response
        )

        # Initialize the results list
        results = []

        # Evaluate using each LLM
        for model_name in eval_models:
            model = OllamaLLM(model=model_name)
            evaluation_results_str = model.invoke(prompt)
            evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

            logger.info(f"Evaluation Prompt for {model_name}:\n{prompt}")

            # Check if the evaluation result is 'true' or 'false'
            if "true" in evaluation_results_str_cleaned:
                logger.info("\033[92m" + f"Response from {model_name}: {evaluation_results_str_cleaned}" + "\033[0m")
                results.append(True)
            elif "false" in evaluation_results_str_cleaned:
                logger.error("\033[91m" + f"Response from {model_name}: {evaluation_results_str_cleaned}" + "\033[0m")
                results.append(False)
            else:
                raise ValueError(
                    f"Invalid evaluation result from {model_name}. Cannot determine if 'true' or 'false'."
                )

        # If only one model is provided, return its result directly
        if len(eval_models) == 1:
            return results[0]

        # If three models are provided, implement voting mechanism
        elif len(eval_models) == 3:
            true_count = results.count(True)
            false_count = results.count(False)

            if true_count >= 2:
                logger.info("\033[92m" + "Majority voting result: True" + "\033[0m")
                return True
            elif false_count >= 2:
                logger.error("\033[91m" + "Majority voting result: False" + "\033[0m")
                return False
            else:
                raise ValueError(
                    "No clear majority in voting. Cannot determine if 'true' or 'false'."
                )

        # Handle cases where the number of models is not 1 or 3
        else:
            raise ValueError(
                "The number of evaluation models must be either 1 or 3."
            )

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        return False


def evaluate_questions(json_file: str, output_file: str = "eval_results/evaluation_results.json", rag_model='mistral', eval_models=['mistral']):
    """
    Evaluate all questions in the JSON file using the RAG model and Ollama LLM.
    If a `rag_response` already exists in the JSON file, it will be used instead of generating a new one.

    Args:
        json_file (str): Path to the JSON file containing questions and expected responses.
        output_file (str): Path to save the evaluation results.
        rag_model (str): The RAG model to use for generating responses (if needed).
        eval_models (list): List of evaluation models to use for judging correctness.
    """
    # Load questions
    questions = load_questions(json_file)
    if not questions:
        return

    # Store evaluation results
    evaluation_results = []

    # Evaluate each question
    for i, q in enumerate(questions):
        question_text = q.get("question")
        expected_response = q.get("expected_response")
        rag_response = q.get("rag_response")  # Check if `rag_response` already exists

        if not question_text or not expected_response:
            logger.warning(f"Skipping question {i + 1} due to missing 'question' or 'expected_response'.")
            continue

        logger.info(f"Evaluating question {i + 1}: {question_text}")

        # If `rag_response` is not already in the JSON, generate it
        if not rag_response:
            #logger.info(f"Generating RAG response for question {i + 1}.")
            rag_response, sources = query_rag_nested(
                query_text=question_text,
                model=rag_model,
                emb_model="nomic-embed-text"
            )
        # else:
        #     logger.info(f"Using existing RAG response for question {i + 1}.")

        # Evaluate the response
        is_correct = evaluate_response(expected_response, rag_response, eval_models)

        # Store the result
        result = {
            "question_number": i + 1,
            "question": question_text,
            "expected_response": expected_response,
            "rag_response": rag_response,
            # "sources": sources,  # Uncomment if you want to include sources
            "is_correct": is_correct
        }
        evaluation_results.append(result)

        logger.info(f"Question {i + 1} evaluation complete. Correct: {is_correct}")

    # Save evaluation results to a JSON file
    try:
        with open(output_file, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        logger.info(f"Evaluation results saved to {output_file}.")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")


def calculate_accuracy(evaluation_file: str) -> float:
    """
    Calculates the accuracy from the evaluation results JSON file.

    Args:
        evaluation_file (str): Path to the evaluation results JSON file.

    Returns:
        float: The accuracy as a percentage (0.0 to 100.0).  Returns 0.0 if no results are found or if there's an error.
    """
    try:
        with open(evaluation_file, "r") as f:
            evaluation_results = json.load(f)

        if not evaluation_results:
            logger.warning("No evaluation results found in the JSON file.")
            return 0.0

        correct_count = sum(1 for result in evaluation_results if result.get("is_correct") is True)
        total_count = len(evaluation_results)

        accuracy = (correct_count / total_count) 
        logger.info(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    except FileNotFoundError:
        logger.error(f"Evaluation file not found: {evaluation_file}")
        return 0.0
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {evaluation_file}")
        return 0.0
    except Exception as e:
        logger.error(f"An error occurred while calculating accuracy: {e}")
        return 0.0




    
def evaluate_llm_similarity(json_file: str, model_name: str = "llama2", output_file: str = "evaluation_results.json"):
    """
    Evaluates a language model's ability to determine semantic similarity between two statements
    based on data in a JSON file.

    Args:
        json_file: Path to the JSON file containing statement pairs and expected similarity.
        model_name: The name of the language model to use.  Defaults to "llama2".
        output_file: The desired name for the output JSON file. Defaults to "evaluation_results.json".

    Returns:
        Path to a new JSON file containing the evaluation results, including predicted similarity.
    """

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {json_file}")
        return None

    results = []
    model_llm = OllamaLLM(model=model_name)

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an expert in determining if two statements have a similar meaning. 
        Respond with only "true" or "false".  Be strict - only respond "true" if the statements convey *essentially* the same information.
        
        Statement 1: {statement1}
        Statement 2: {statement2}
        
        Do these statements have a similar meaning? (true/false):
        """
    )

    for example in data:
        statement1 = example.get("statement1", "")
        statement2 = example.get("statement2", "")
        expected_similarity = example.get("similarity", None)  #Crucially get the expected value

        prompt = prompt_template.format(statement1=statement1, statement2=statement2)
        llm_response = model_llm.invoke(prompt).strip().lower()  # Normalize the response

        predicted_similarity = 1 if "true" in llm_response else 0

        result = {
            "example_number": example.get("example_number"),
            "expected_similarity": expected_similarity,
            "predicted_similarity": predicted_similarity
        }
        results.append(result)

    # Save the results to a new JSON file
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Evaluation complete. Results saved to: {output_file}")
    return output_file

def evaluate_results_file(output_file: str):
    """
    Evaluates the results stored in a JSON file (created by evaluate_llm_similarity).
    Calculates and prints evaluation metrics (accuracy, precision, recall, F1, TP, TN, FP, FN).

    Args:
        output_file: Path to the JSON file containing evaluation results 
                     (with 'expected_similarity' and 'predicted_similarity' fields).

    Returns:
        None. Prints evaluation metrics to the console.
    """
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {output_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {output_file}")
        return

    y_true = []
    y_pred = []

    for entry in data:
        expected_similarity = entry.get("expected_similarity")
        predicted_similarity = entry.get("predicted_similarity")

        if expected_similarity is not None and predicted_similarity is not None:
            y_true.append(expected_similarity)
            y_pred.append(predicted_similarity)

    if len(y_true) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }

        print(f"Evaluation Metrics (from {output_file}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  True Positives (TP): {tp}")
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")

        return metrics


    else:
        print(f"Warning: No examples with valid 'expected_similarity' and 'predicted_similarity' values found in {output_file}. Cannot calculate evaluation metrics.")
    
