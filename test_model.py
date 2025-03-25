from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import os
import json
import argparse

def load_model(model_path):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    # Load base model with default parameters
    base_model_name = "unsloth/llama-3-8b-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )
    
    # Create PEFT model configuration
    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        r=16,
    )
    
    # Load the adapter weights
    print("Loading adapter weights...")
    try:
        # First try to load the adapter config
        from peft import PeftConfig
        config = PeftConfig.from_pretrained(model_path)
        print(f"Loaded adapter config with target modules: {config.target_modules}")
        
        # Then load the weights
        model = PeftModel.from_pretrained(model, model_path)
        print("Successfully loaded adapter weights")
    except Exception as e:
        print(f"Warning: Could not load adapter config: {e}")
        print("Attempting to load weights directly...")
        model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
    """Generate a response for a given prompt."""
    # Format the prompt according to the training template
    formatted_prompt = f"""Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{prompt}

### Response:"""
    
    # Generate response with attention mask
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True
    ).to("cuda")
    
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the response part (after "### Response:")
    response = response.split("### Response:")[-1].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to test the model with")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(args.model_path)
        
        # Generate response
        response = generate_response(
            model, 
            tokenizer, 
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print("\nPrompt:")
        print(args.prompt)
        print("\nResponse:")
        print(response)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
