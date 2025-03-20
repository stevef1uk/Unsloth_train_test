from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import time
import sys
import os
import threading

# Add debug prints
print("Script started")


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
print(f"Contents: {os.listdir(cache_dir)}")
model_name = "unsloth/llama-3-8b-bnb-4bit"
print(f"Checking if model files exist in cache at {cache_dir}")
if os.path.exists(cache_dir):
    possible_model_dirs = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir)]
    for d in possible_model_dirs:
        if model_name.split("/")[-1] in d:
            print(f"Found potential model directory: {d}")
            files = os.listdir(d)
            print(f"Files in directory: {files}")

max_seq_length = 2048  # Supports RoPE Scaling internally

# Try a different approach - let's try to load the model directly from the Hugging Face transformers library
print("Trying alternative loading approach...")
sys.stdout.flush()

try:
    print("Starting model download with alternative method...")
    sys.stdout.flush()
    
    # Set a timeout for the model loading
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Model loading timed out")
    
    # Set 10-minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)
    
    # Try loading with different parameters
    print("Attempting to load model with specific parameters...")
    sys.stdout.flush()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        device_map = "auto",  # This might help with loading
        attn_implementation = "flash_attention_2",  # Try a different attention implementation
        local_files_only = False,  # Try to use cached files only
    )
    
    print("Model loaded successfully!")
    signal.alarm(0)  # Cancel the timeout
   

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = False, # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )


 
    # If we get here, the model loaded successfully
    print("Continuing with training setup...")
    
    # Rest of your code...

    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
    print(dataset.column_names)


    from unsloth import to_sharegpt

    dataset = to_sharegpt(
        dataset,
        merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
        output_column_name="output",
        conversation_extension=3,  # Select more to handle longer conversations
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
        # default_system_message = "You are a helpful assistant", << [OPTIONAL]
    )

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        tokenizer = tokenizer,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            per_device_train_batch_size = 10,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 60,
            logging_steps = 1,
            output_dir = "outputs",
            optim = "adamw_8bit",
            seed = 3407,
        ),
    )
    trainer.train()

    # Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
    # (1) Saving to GGUF / merging to 16bit for vLLM
    # (2) Continued training from a saved LoRA adapter
    # (3) Adding an evaluation loop / OOMs
    # (4) Customized chat templates


    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")



except TimeoutError as e:
    print(f"Timeout error: {e}")
    print("Model loading took too long. Examining the state of downloaded files...")
    
    # Try to investigate what's happening with the cache
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

