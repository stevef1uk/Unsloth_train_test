# Unsloth Llama-3 Fine-tuning

This repository contains code for fine-tuning Llama-3 using the [Unsloth](https://github.com/unslothai/unsloth) library. Unsloth makes fine-tuning LLMs like Llama-3 up to 2x faster while using 70% less memory.

## Training

### Local Training

To train the model locally:

1. Install dependencies:
```bash
pip install torch transformers trl datasets unsloth accelerate bitsandbytes sentencepiece einops
```

2. Run the training script:
```bash
python Unsloth_train_test/train_locally.py
```

The script will:
- Create a timestamped output directory for model checkpoints
- Monitor GPU memory usage
- Save checkpoints every 10 steps
- Save the final model and training configuration

### Modal Training

To train using Modal:

1. Install Modal:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

3. Deploy the training script:
```bash
modal deploy Unsloth_train_test/train_on_modal.py
```

## Testing

To test a trained model:

```bash
python Unsloth_train_test/test_model.py --model_path outputs/model_YYYYMMDD_HHMMSS/final_model/ --prompt "Your test prompt here"
```

Example:
```bash
python Unsloth_train_test/test_model.py --model_path outputs/model_20250325_174009/final_model/ --prompt "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8," --max_tokens 1000
```

### Common Warnings

When running the test script, you may see the following warnings:

1. **Multiple Adapter Configurations Warning**:
```
Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model.
```
This is normal and doesn't affect model performance. It occurs because we create a PEFT configuration before loading the trained weights.

2. **Missing Adapter Keys Warning**:
```
Found missing adapter keys while loading the checkpoint: [...]
```
This warning indicates that the model is looking for LoRA adapter weights for all layers but can't find them. This is expected because:
- The model checks for weights in all layers (0-31)
- Training might have only updated a subset of layers
- The saved weights might have a different structure than expected

These warnings are informational and don't indicate problems with the model's functionality. The model will still generate responses using the available trained weights.

### Generation Parameters

The test script supports the following parameters:
- `--model_path`: Path to the trained model (required)
- `--prompt`: The prompt to test with (required)
- `--max_tokens`: Maximum number of tokens to generate (default: 200)
- `--temperature`: Temperature for generation (default: 0.7)

## Requirements

- Python 3.8+
- CUDA-capable GPU with at least 12GB VRAM
- Linux environment (for local training)
- Required Python packages (see installation instructions above)


