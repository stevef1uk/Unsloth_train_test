# Unsloth Llama-3 Fine-tuning

This repository contains code for fine-tuning Llama-3 using the [Unsloth](https://github.com/unslothai/unsloth) library. Unsloth makes fine-tuning LLMs like Llama-3 up to 2x faster while using 70% less memory.

## Overview

This repository contains two main scripts:

1. `train_on_modal.py` - For running training in the cloud using Modal's GPU infrastructure
2. `train_locally.py` - For running training on a local machine with a GPU

Both scripts implement the same training pipeline for fine-tuning Llama-3 on the Alpaca dataset, but they're configured for different environments.

## Script 1: train_on_modal.py - Cloud Training

The `train_on_modal.py` script is designed to leverage Modal's cloud infrastructure for training. It includes:

- Setting up Modal volumes for caching models and storing results
- Creating a container image with all dependencies
- Loading and configuring the Llama-3 model
- Preparing the Alpaca dataset
- Training using LoRA (Low-Rank Adaptation)
- Saving and testing the fine-tuned model

### Running on Modal

#### Prerequisites

1. Install Modal CLI:
   ```
   pip install modal
   ```

2. Authenticate with Modal:
   ```
   modal token new
   ```

#### Deployment and Training

1. Deploy the script to Modal:
   ```
   modal deploy Unsloth_train_test/train_on_modal.py
   ```

2. Run the training function:
   ```
   modal run Unsloth_train_test/train_on_modal.py::train
   ```

3. After training completes, check the trained model:
   ```
   modal run Unsloth_train_test/train_on_modal.py::check_model
   ```

### Key Components of train_on_modal.py

- **Modal App Setup**: Creates a Modal app and volumes for caching and saving models
- **Container Image**: Builds an image with CUDA 11.8, PyTorch, and ML libraries
- **Training Function**: `@app.function` decorator runs the training on a T4 GPU
- **Checkpoint Saving**: Saves checkpoints every 5 steps to handle potential preemption
- **Model Testing**: Includes a separate function for testing the model after training

## Script 2: train_locally.py - Local Training

The `train_locally.py` script is essentially the same as `train_on_modal.py` but configured for local execution. It contains Modal-specific code but can be adapted to run on your local machine with a GPU.

### Running Locally

To run this fine-tuning process locally, you'll need a machine with a GPU that has at least 16GB VRAM.

#### Prerequisites

1. Install the required dependencies:

```bash
# Install PyTorch with CUDA support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install ML dependencies
pip install unsloth transformers>=4.34.0 trl>=0.7.4 datasets accelerate>=0.23.0 bitsandbytes>=0.41.2 sentencepiece einops
```

2. Adapt the `train_locally.py` file to run locally by:
   - Removing Modal-specific imports and decorators
   - Setting up local paths for model and checkpoint storage
   - Adjusting batch size based on your GPU memory

#### Creating a Pure Local Version

You can convert `train_locally.py` to a pure local script by:

1. Removing all Modal-specific code:
   - `import modal` and Modal app/volume/image definitions
   - `@app.function` decorators

2. Replacing Modal paths with local paths:
   - Change `/root/.cache/huggingface` to a local path
   - Change `/outputs/unsloth-model` to a local directory

3. Directly calling the training function in the `__main__` block:
   ```python
   if __name__ == "__main__":
       train()
   ```

### Training Parameters

Both scripts use the following key parameters:

- **Model**: `unsloth/llama-3-8b-bnb-4bit` (8B parameter model in 4-bit quantization)
- **Max sequence length**: 2048 tokens
- **LoRA rank**: 16
- **Batch size**: 4 per device
- **Gradient accumulation steps**: 4
- **Training steps**: 30
- **Checkpointing**: Every 5 steps

## Exporting to Ollama

After fine-tuning, you can export your model to [Ollama](https://ollama.ai/) to run it locally on your machine.

### Steps to Export to Ollama:

1. Install Ollama on your machine following [Ollama's installation guide](https://github.com/ollama/ollama#installation)

2. Export your fine-tuned model to GGUF format. You can use the following libraries:
   - [llama.cpp](https://github.com/ggerganov/llama.cpp)
   - Unsloth's export utilities

3. Create a Modelfile for Ollama with the following content:
   ```
   FROM ./path/to/your/exported/model.gguf
   
   TEMPLATE """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

   ### Instruction:
   {{.Input}}

   ### Response:
   """
   
   PARAMETER temperature 0.2
   PARAMETER top_p 0.95
   ```

4. Create the Ollama model:
   ```
   ollama create unsloth_model -f Modelfile
   ```

5. Run your model:
   ```
   ollama run unsloth_model
   ```

For more detailed information about the fine-tuning process with Unsloth and exporting to Ollama, refer to the [Unsloth documentation](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-6.-alpaca-dataset).

## Differences Between the Scripts

While both scripts contain nearly identical code, they serve different purposes:

1. **train_on_modal.py**: 
   - Designed to run in Modal's cloud environment
   - Uses Modal volumes for persistence
   - Configured to leverage Modal's GPU resources
   - Handles preemption through checkpointing

2. **train_locally.py**:
   - Can be modified to run on a local GPU
   - Contains the same Modal code but needs adaptation for local use
   - Serves as a template for local execution

## Additional Resources

- [Unsloth GitHub Repository](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Modal Documentation](https://modal.com/docs/guide)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce `per_device_train_batch_size` or increase the quantization level
2. **Modal Preemption**: The training will automatically resume from the last checkpoint
3. **Missing Libraries**: Make sure all required packages are installed with the correct versions
4. **Local GPU Not Detected**: Ensure CUDA is properly installed and PyTorch can access your GPU


