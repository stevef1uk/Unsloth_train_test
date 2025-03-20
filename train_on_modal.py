import sys
import os
import modal

# Create Modal app (updated from Stub)
app = modal.App("unsloth-training")

# Create volumes for model cache and output
cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
model_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

# Create Modal image with all required dependencies
# Use Modal's CUDA image as a base instead of debian_slim
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",  # Using a known working version
        add_python="3.10"
    )
    # Install system dependencies including build tools and C compiler
    .apt_install(
        "build-essential", 
        "gcc", 
        "g++", 
        "git",
        "cmake",
        "pkg-config"
    )
    # Install PyTorch with CUDA support - split into separate commands
    .pip_install("torch==2.1.2", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("torchvision==0.16.2", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("torchaudio==2.1.2", index_url="https://download.pytorch.org/whl/cu118")
    # Install other ML dependencies
    .pip_install(
        "unsloth", 
        "transformers>=4.34.0", 
        "trl>=0.7.4", 
        "datasets", 
        "accelerate>=0.23.0", 
        "bitsandbytes>=0.41.2", 
        "sentencepiece", 
        "einops",
    )
)

# Main function to run the training
@app.function(
    image=image,
    gpu="T4", # Options: "T4", "L4", "A10G", "A100", "A100-80GB", "H100"
    timeout=7200,  # Increased to 2 hours (7200 seconds) from 1 hour
    volumes={
        "/root/.cache/huggingface": cache_volume,
        "/outputs": model_volume
    }
)
def train():
    # Import packages inside the function
    # This way they're only imported when running on Modal
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    import time
    import threading
    
    # Add debug prints
    print("Script started on Modal")

    # Function to monitor GPU memory usage
    def monitor_gpu():
        try:
            import subprocess
            while True:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], 
                                capture_output=True, text=True)
                print(f"GPU Memory: {result.stdout.strip()}")
                sys.stdout.flush()
                time.sleep(5)
        except Exception as e:
            print(f"Error monitoring GPU: {e}")

    # Start GPU monitoring in a separate thread
    gpu_thread = threading.Thread(target=monitor_gpu, daemon=True)
    gpu_thread.start()

    # Check if the model is already cached
    cache_dir = "/root/.cache/huggingface/hub"
    print(f"Cache directory: {cache_dir}")
    if os.path.exists(cache_dir):
        print(f"Contents: {os.listdir(cache_dir)}")
    
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length = 2048  # Supports RoPE Scaling internally

    print("Starting model download...")
    sys.stdout.flush()
    
    try:
        # Load the model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        
        print("Model loaded successfully!")
       
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        print("Continuing with training setup...")
        
        # Load dataset
        dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
        print(f"Dataset loaded: {dataset.column_names}")

        from unsloth import to_sharegpt
        dataset = to_sharegpt(
            dataset,
            merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
            output_column_name="output",
            conversation_extension=3,
        )

        from unsloth import standardize_sharegpt
        dataset = standardize_sharegpt(dataset)
        
        chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.
        
        ### Instruction:
        {INPUT}
        
        ### Response:
        {OUTPUT}"""
        
        from unsloth import apply_chat_template
        dataset = apply_chat_template(
            dataset,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )

        from trl import SFTTrainer
        from transformers import TrainingArguments

        output_dir = "/outputs/unsloth-model"
        
        # Configure the trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=SFTConfig(
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=30,
                logging_steps=1,
                save_steps=5,
                save_total_limit=3,
                output_dir=output_dir,
                optim="adamw_8bit",
                seed=3407,
            ),
        )
        
        # Train the model
        trainer.train()

        # Save the trained model to the volume
        model.save_pretrained(f"{output_dir}/lora_model")
        tokenizer.save_pretrained(f"{output_dir}/lora_model")
        
        print(f"Training complete! Model saved to {output_dir}/lora_model")
        return {"status": "success", "model_path": f"{output_dir}/lora_model"}

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

# Function to check the trained model
@app.function(
    image=image,
    gpu="T4",
    timeout=1200,
    volumes={
        "/outputs": model_volume
    }
)
def check_model():
    # Import libraries inside the function
    from unsloth import FastLanguageModel
    import os
    from peft import PeftModel
    
    try:
        print("Checking for trained model in volume...")
        model_path = "/outputs/unsloth-model/lora_model"
        
        if not os.path.exists(model_path):
            return {"status": "error", "message": f"Model path {model_path} not found"}
        
        print(f"Model directory exists. Contents: {os.listdir(model_path)}")
        
        # Try to load the model to verify it's valid
        print("Attempting to load the trained model...")
        
        # Load base model first
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
        
        # Now load the adapter weights
        print(f"Loading adapter weights from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path)
        
        # Test generation with Fibonacci sequence prompt
        prompt = "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"
        print(f"Running test generation with prompt: '{prompt}'")
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,  # Shorter sequence for this task
            temperature=0.2,    # Lower temperature for more deterministic output
            top_p=0.95,
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated text:\n{generated_text}")
        
        return {
            "status": "success", 
            "message": "Model loaded and tested successfully",
            "sample_output": generated_text
        }
        
    except Exception as e:
        import traceback
        print(f"Error checking model: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

# For local running without Modal
if __name__ == "__main__":
    print("To run training on Modal: modal run test.py::train")
    print("To check trained model: modal run test.py::check_model")
    print("To deploy on Modal: modal deploy test.py")

