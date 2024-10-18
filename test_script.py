from openai import OpenAI
import os
from codebleu.codebleu import calc_codebleu
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import anthropic

def generate_one_completion_gpt(prompt):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    system_prompt = "You are a helpful coding assistant. Write clean, functional code. Just the code without any explanation or ticks"
    
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    return completion.choices[0].message.content

def generate_one_completion_llama(prompt):
    # Load the LLaMA model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")  # Example model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=300, do_sample=True, temperature=0.7)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

def generate_one_completion_claude(prompt):
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    client = anthropic.Client(api_key=claude_api_key)
    
    response = client.completions.create(
        model="claude-v1",  # Use the appropriate model (e.g., claude-v1, claude-v2)
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
        max_tokens_to_sample=300,
        temperature=0.7,
    )
    
    generated_code = response['completion'].strip()
    return generated_code

def generate_one_completion(prompt, model="gpt"):
    if model == "gpt":
        return generate_one_completion_gpt(prompt)
    elif model == "llama":
        return generate_one_completion_llama(prompt)
    elif model == "claude":
        return generate_one_completion_claude(prompt)
    else:
        raise ValueError("Unsupported model")

if __name__ == "__main__":
    prompt = "Write a Python function that calculates the factorial of a number."
    generated_code = generate_one_completion(prompt)
    print("Generated Code: ", generated_code)

    references = ["def factorial(n): return 1 if n == 0 else n * factorial(n - 1)"]

    predictions = [generated_code]

    result = calc_codebleu(references, predictions, "python")

    print("CodeBLEU result:")
    print(result)