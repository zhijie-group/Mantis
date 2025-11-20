import json
from openai import OpenAI

client = OpenAI(
   api_key="sk-38986cf55ddc4b67a391dcf7bf0ea627",
   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def load_data_json(file_path):
   """Load the question data from JSON file"""
   with open(file_path, 'r', encoding='utf-8') as f:
       return json.load(f)

def load_results_json(file_path):
   """Load model results from JSON file"""
   results = {}
   with open(file_path, 'r', encoding='utf-8') as f:
       data = json.load(f)
       for item in data:
           results[item['id']] = item['output']
   return results

def check_answer_consistency(question, standard_answer, model_answer):
   """Check if model answer is consistent with standard answer using LLM"""
   prompt = f"""You are an answer evaluation assistant. Please determine whether the following two answers are semantically consistent for the given question.
       Question: {question}

       Standard Answer: {standard_answer}
       Model Answer: {model_answer}

       Evaluation Rules:
       1. Case-insensitive (Yes and yes are considered the same)
       2. For affirmative answers: Yes, yes, true, True, 1, correct are all considered consistent
       3. For negative answers: No, no, false, False, 0, 2, incorrect are all considered consistent
       4. For option answers (such as A, B, C): Same letter is sufficient, ignore extra symbols (e.g., "A" and "A." are considered the same)
       5. Ignore extra punctuation, spaces, and formatting differences
       6. Focus only on whether the core semantics are consistent with the question context

       Please respond with only "Correct" or "Wrong", with no other content.
   """

   completion = client.chat.completions.create(
       model="qwen-plus",
       messages=[
           {"role": "system", "content": "You are a helpful assistant for answer evaluation."},
           {"role": "user", "content": prompt},
       ],
   )
   
   answer = json.loads(completion.model_dump_json())
   result = answer["choices"][0]["message"]["content"].strip()
   
   return "Correct" in result or "correct" in result

def evaluate_results(data_json_path, results_json_path):
   """
   Evaluate results and generate statistical report
   
   Args:
       data_json_path: Path to the question data JSON file
       results_json_path: Path to the model results JSON file
   
   Returns:
       tuple: (correct_count, total_count)
   """
   # Load data
   data = load_data_json(data_json_path)
   model_results = load_results_json(results_json_path)

   total = len(data)
   correct = 0
   wrong = 0
   details = []
   
   print("Starting evaluation...\n")
   print("=" * 80)

   for item in data:
       question_id = item['id']
       standard_answer = item['answer']
       question = item['question']

       # Get model answer from results
       model_answer = model_results.get(question_id, "")

       # Check consistency
       is_consistent = check_answer_consistency(question, standard_answer, model_answer)

       if is_consistent:
           correct += 1
           status = "✓ Correct"
       else:
           wrong += 1
           status = "✗ Wrong"

       # Store detail
       detail = {
           "id": question_id,
           "question": question[:100] + "..." if len(question) > 100 else question,
           "standard_answer": standard_answer,
           "model_answer": model_answer,
           "status": status
       }
       details.append(detail)

       # Print progress
       print(f"ID: {question_id} | Standard: {standard_answer} | Model: {model_answer} | {status}")

   # Print summary
   print("\n" + "=" * 80)
   print("\nEvaluation Report:")
   print(f"Total Questions: {total}")
   print(f"Correct: {correct}")
   print(f"Wrong: {wrong}")
   print(f"Accuracy: {correct/total*100:.2f}%")
   print("\n" + "=" * 80)

   return correct, total

if __name__ == "__main__":

   data_json_path = "/data/yangyi/datasets/language_eval/realworldqa/realworldqa.json"
   results_json_path = "/data/yangyi/metaquery_action_refactoring/eval_language/realworldqa/results/mantis_results_20251106_052040.json"

   correct, total = evaluate_results(data_json_path, results_json_path)
