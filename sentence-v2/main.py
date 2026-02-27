import json
import pandas as pd
import os
import random
from mlx_lm import load, generate

# 1. SETUP
MODEL_PATH = "mlx-community/Qwen2.5-14B-Instruct-4bit"
DATASET_FILE = "./data/ROCStories_winter2017.csv" 
OUTPUT_FILE = "./data/asl_variable_length_dataset.json"

model, tokenizer = load(MODEL_PATH)

def get_asl_prompt(sentences):
    text_block = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
    return (
        "TASK: Convert the following English sentences into ASL Gloss.\n"
        "RULES: No articles, No 'to be' verbs, Topic-Comment structure.\n"
        "If the input is one sentence, output one gloss. If it's a list, output a list.\n\n"
        f"ENGLISH:\n{text_block}\n\nASL GLOSSES:"
    )

# 2. LOAD & FLATTEN DATA
df = pd.read_csv(DATASET_FILE)
all_sentences = []
for _, row in df.iterrows():
    all_sentences.extend([row['sentence1'], row['sentence2'], row['sentence3'], row['sentence4'], row['sentence5']])

# 3. VARIABLE LENGTH PROCESSING
dataset = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        dataset = json.load(f)

# Track how many sentences we've already processed
current_pos = len(dataset)

while current_pos < len(all_sentences):
    # Randomly choose a chunk size between 1 and 6
    chunk_size = random.randint(1, 6)
    batch = all_sentences[current_pos : current_pos + chunk_size]
    
    if not batch: break

    prompt = get_asl_prompt(batch)
    
    try:
        # Generate with higher temperature for variety
        response = generate(model, tokenizer, prompt=prompt, max_tokens=1000, verbose=False, temp=0.7)
        
        # Clean response
        lines = [line.split('. ', 1)[-1].strip() for line in response.strip().split('\n') if line.strip()]
        
        # Ensure we don't save more glosses than we have sentences
        for eng, gloss in zip(batch, lines):
            dataset.append({"sentence": eng, "gloss": gloss})
            
        current_pos += len(batch)

        if len(dataset) % 100 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(dataset, f, indent=4)
            print(f"Generated {len(dataset)} pairs. Current chunk size: {chunk_size}")

    except Exception as e:
        print(f"Error: {e}")
        current_pos += 1 # Skip problematic sentence

# Final save
with open(OUTPUT_FILE, "w") as f:
    json.dump(dataset, f, indent=4)
print("Done! Dataset has variable sentence context.")