import json
import os

def preprocess_dsa_data(input_file, output_file):
    """
    Specialized preprocessor for DSA content.
    Formats data to emphasize Problem-Solving, Code, and Complexity.
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return 0

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by common headers or large gaps if any
    # For now, we take chunks to create training pairs
    lines = content.split('\n')
    processed_data = []
    
    # Simple chunking for demo purposes - in a real scenario, this would be more semantic
    chunk_size = 15
    for i in range(0, len(lines), chunk_size):
        chunk = " ".join(lines[i:i + chunk_size])
        if len(chunk) < 100:
            continue
            
        # Format for DSA Assistant
        text = (
            f"<|im_start|>system\nYou are an expert DSA tutor. Explain algorithms and code with complexity analysis.<|im_end|>\n"
            f"<|im_start|>user\nAnalyze this DSA concept and provide a summary: {chunk[:500]}...<|im_end|>\n"
            f"<|im_start|>assistant\nHere is a detailed breakdown of the DSA concept:\n\n{chunk}<|im_end|>"
        )
        processed_data.append({"text": text})

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Created {len(processed_data)} DSA training examples in {output_file}")
    return len(processed_data)

if __name__ == "__main__":
    preprocess_dsa_data("data/dsa_raw.txt", "data/dsa_processed.jsonl")
