#!/usr/bin/env python3
"""
This script tunes hyperparameters for the Ollama LLM "deepseek-r1:14b" using Bayesian optimization.
The hyperparameters being tuned are:
    - temperature: controls randomness of predictions.
    - max_tokens: maximum number of tokens in the output.
    - top_k: limits the number of tokens considered at each generation step.
    - top_p: nucleus sampling threshold.
    - repetition_penalty: penalizes repeated tokens.

The objective is to maximize the similarity between the LLM’s generated output for fixed prompts
and their target strings. When a candidate configuration yields a new best similarity, a checkpoint is saved.
After up to 30 iterations, the convergence is plotted and a report is saved to bayes_opt.txt.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import difflib
import GPyOpt
import ollama

# Global variable to track the best similarity seen so far.
global_best_similarity = 0.0


def call_llm(prompt, temperature, max_tokens, top_k, top_p, repetition_penalty):
    """
    Calls the deepseek-r1:14b LLM via Ollama with the specified hyperparameters.

    Since the current Ollama API does not accept a 'parameters' argument in the chat() call,
    we create a temporary model with the desired hyperparameters using ollama.create.

    Parameters:
        prompt (str): The prompt to send to the LLM.
        temperature (float): Controls randomness (lower is more deterministic).
        max_tokens (float): Maximum tokens to generate.
        top_k (float): Number of tokens considered at each step.
        top_p (float): Nucleus sampling threshold.
        repetition_penalty (float): Penalty for repeated tokens.

    Returns:
        str: The generated text.
    """
    # Create a configuration string with the desired hyperparameters.
    # (We omit the FROM line because we pass it separately.)
    config = f"""
PARAMETER temperature {temperature}
PARAMETER max_tokens {int(round(max_tokens))}
PARAMETER top_k {int(round(top_k))}
PARAMETER top_p {top_p}
PARAMETER repetition_penalty {repetition_penalty}
""".strip()
    # Create (or update) a temporary model using the base model "deepseek-r1:14b"
    # and the configuration specified in the template.
    ollama.create("temp_model", from_="deepseek-r1:32b", template=config)

    # Call the chat API using the temporary model.
    response = ollama.chat(
        model="temp_model",
        messages=[
            {"role": "system",
             "content": "Ignore the internal configuration details and focus solely on the user prompt."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']


# Load prompt-target pairs from a JSON file.
with open('prompts_targets.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

examples = data['examples']


def compute_overall_similarity(generated_texts, target_texts):
    """
    Computes the average similarity over multiple prompt-target pairs.
    """
    total_similarity = 0.0
    for gen, target in zip(generated_texts, target_texts):
        total_similarity += difflib.SequenceMatcher(None, gen, target).ratio()
    return total_similarity / len(generated_texts)


def objective_function(params):
    """
    Objective function for Bayesian optimization.

    For each candidate hyperparameter set, calls the LLM for each prompt from the JSON file,
    computes the average similarity with the target texts, and returns the negative similarity.

    Additionally, prints the hyperparameters, prompts, generated responses, and similarity
    for inspection during optimization.

    Parameters:
        params (np.ndarray): Each row contains:
            [temperature, max_tokens, top_k, top_p, repetition_penalty]

    Returns:
        np.ndarray: Negative similarity scores (shape: (n, 1)).
    """
    global global_best_similarity
    results = []
    # Process each candidate hyperparameter set.
    for idx, x in enumerate(params):
        temperature = float(x[0])
        max_tokens = float(x[1])
        top_k = float(x[2])
        top_p = float(x[3])
        repetition_penalty = float(x[4])

        generated_texts = []
        target_texts = []

        # print(f"\n--- Evaluating Candidate {idx + 1} ---")
        # print(f"Hyperparameters: temperature={temperature:.3f}, max_tokens={int(round(max_tokens))}, "
        #     f"top_k={int(round(top_k))}, top_p={top_p:.3f}, repetition_penalty={repetition_penalty:.3f}\n")

        # Evaluate each prompt-target pair.
        for example in examples:
            prompt = example["prompt"]
            target = example["target_text"]
            target_texts.append(target)

            try:
                generated = call_llm(prompt, temperature, max_tokens, top_k, top_p, repetition_penalty)
            except Exception as e:
                print(f"Error calling LLM: {e}")
                generated = ""

            filtered_generated = "\n".join(
                line for line in generated.splitlines() if not line.strip().startswith("PARAMETER")
            )

            generated_texts.append(filtered_generated)

            # Print details for this example.
            print(f"Prompt: {prompt}")
            print(f"Generated: {filtered_generated}\n")

        similarity = compute_overall_similarity(generated_texts, target_texts)

        if similarity > global_best_similarity:
            global_best_similarity = similarity
            checkpoint_filename = (
                f"checkpoint_temp-{temperature:.3f}_tokens-{int(round(max_tokens))}_"
                f"topk-{int(round(top_k))}_topp-{top_p:.3f}_repp-{repetition_penalty:.3f}.txt"
            )
            with open(checkpoint_filename, "w") as f:
                f.write("Hyperparameters:\n")
                f.write(f"  Temperature: {temperature:.3f}\n")
                f.write(f"  Max Tokens: {int(round(max_tokens))}\n")
                f.write(f"  Top_k: {int(round(top_k))}\n")
                f.write(f"  Top_p: {top_p:.3f}\n")
                f.write(f"  Repetition Penalty: {repetition_penalty:.3f}\n\n")
                for i, example in enumerate(examples):
                    f.write(f"Prompt: {example['prompt']}\n")
                    f.write(f"Target: {example['target_text']}\n")
                    f.write(f"Generated: {generated_texts[i]}\n")
                    f.write("-" * 20 + "\n")
                f.write(f"Average Similarity: {similarity:.4f}\n")

        # Append negative similarity (since GPyOpt minimizes the objective).
        results.append(-similarity)

    return np.array(results).reshape(-1, 1)


# Define the search domain for the five hyperparameters.
domain = [
    {'name': 'temperature', 'type': 'continuous', 'domain': (0.1, 1.0)},
    {'name': 'max_tokens', 'type': 'continuous', 'domain': (16, 256)},
    {'name': 'top_k', 'type': 'continuous', 'domain': (0, 200)},
    {'name': 'top_p', 'type': 'continuous', 'domain': (0.0, 1.0)},
    {'name': 'repetition_penalty', 'type': 'continuous', 'domain': (1.0, 2.0)}
]

# Set up Bayesian Optimization with an initial design of 5 points.
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=domain,
    initial_design_numdata=5,
    acquisition_type='EI'
)

max_iter = 30
optimizer.run_optimization(max_iter=max_iter, verbosity=True)

# Retrieve best hyperparameters (note: we reverse the sign to obtain the best similarity).
best_params = optimizer.x_opt
best_similarity = -optimizer.fx_opt

plt.figure()
optimizer.plot_convergence()
plt.title("LLM Hyperparameter Optimization Convergence")
plt.savefig("convergence_plot.png")
plt.show()

# Write a report of the optimization.
report_lines = []
report_lines.append("Bayesian Optimization Report for LLM Hyperparameters\n")
report_lines.append(f"Total iterations: {len(optimizer.Y)}\n")
report_lines.append("Best hyperparameters found:\n")
report_lines.append(f"  Temperature: {best_params[0]:.3f}\n")
report_lines.append(f"  Max Tokens: {int(round(best_params[1]))}\n")
report_lines.append(f"  Top_k: {int(round(best_params[2]))}\n")
report_lines.append(f"  Top_p: {best_params[3]:.3f}\n")
report_lines.append(f"  Repetition Penalty: {best_params[4]:.3f}\n")
report_lines.append(f"Best Similarity: {best_similarity:.4f}\n\n")
report_lines.append("All iterations:\n")

for i, (params, score) in enumerate(zip(optimizer.X, optimizer.Y)):
    temperature = params[0]
    max_tokens = int(round(params[1]))
    top_k = int(round(params[2]))
    top_p = params[3]
    repetition_penalty = params[4]
    sim = -score[0]
    report_lines.append(
        f"Iteration {i + 1}: temperature={temperature:.3f}, max_tokens={max_tokens}, "
        f"top_k={top_k}, top_p={top_p:.3f}, repetition_penalty={repetition_penalty:.3f}, similarity={sim:.4f}\n"
    )

with open("bayes_opt.txt", "w") as f:
    f.writelines(report_lines)


