from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import time
import sys
import os
import threading
from datetime import datetime

# Add debug prints
print("Script started")

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs/model_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

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
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(f"Cache directory: {cache_dir}")
if os.path.exists(cache_dir):
    print(f"Contents: {os.listdir(cache_dir)}")

model_name = "unsloth/llama-3-8b-bnb-4bit"
max_seq_length = 2048  # Supports RoPE Scaling internally

try:
    print("Starting model download...")
    sys.stdout.flush()
    
    # Set a timeout for the model loading
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Model loading timed out")
    
    # Set 10-minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)
    
    print("Attempting to load model with specific parameters...")
    sys.stdout.flush()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
        local_files_only=False,
    )
    
    print("Model loaded successfully!")
    signal.alarm(0)  # Cancel the timeout

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
    
    # Load and process dataset
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

    # Configure the trainer with checkpointing
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=10,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            logging_steps=1,
            save_steps=10,  # Save every 10 steps
            save_total_limit=3,  # Keep only the last 3 checkpoints
            output_dir=output_dir,
            optim="adamw_8bit",
            seed=3407,
        ),
    )

    # Train the model
    trainer.train()

    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"Training complete! Model saved to {final_model_path}")
    
    # Save training configuration
    config = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "output_dir": output_dir,
        "timestamp": timestamp,
    }
    import json
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

except TimeoutError as e:
    print(f"Timeout error: {e}")
    print("Model loading took too long. Examining the state of downloaded files...")
    
    if os.path.exists(cache_dir):
        model_dirs = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) 
                     if os.path.isdir(os.path.join(cache_dir, d))]
        for d in model_dirs:
            if "llama" in d.lower():
                print(f"Examining Llama model directory: {d}")
                files = os.listdir(d)
                for f in files:
                    fpath = os.path.join(d, f)
                    fsize = os.path.getsize(fpath) / (1024*1024*1024)  # Size in GB
                    print(f"  - {f}: {fsize:.2f} GB")
                    
    print("Suggestions to fix the issue:")
    print("1. Check if your disk has enough free space")
    print("2. Try a smaller model like 'unsloth/llama-3-2b-bnb-4bit'")
    print("3. Try clearing the cache with `rm -rf ~/.cache/huggingface/hub`")
    print("4. Try setting `local_files_only=False` to force re-download")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
