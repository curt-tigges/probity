---
base_model: distilgpt2
library_name: peft
tags:
- sft
- distilgpt2
pipeline_tag: text-generation
---

# laptop_simple_test

SFT fine-tuned model trained on the example_dataset dataset.

## Model Details

- **Model Type**: SFT fine-tuned model
- **Base Model**: distilgpt2
- **Dataset**: example_dataset
- **Training Type**: Single Stage

- **Created**: 2025-06-20T16:54:49.867076
- **Status**: completed

## Training Stages

### Stage 1: SFT

- **Dataset**: example_dataset_sft
- **Starting Model**: distilbert/distilgpt2
- **Learning Rate**: 5e-05
- **Epochs**: 1.0
- **Final Loss**: 3.5828

## Usage

```python
# Load the model
python -m scripts.inference example_dataset/laptop_simple_test
```

## Files

- `adapter_model.safetensors`: LoRA adapter weights
- `model_config.json`: Model configuration  
- `peft_config.json`: PEFT/LoRA configuration
- `run_metadata.json`: Complete training metadata
- `trainer_state.json`: Detailed training logs
