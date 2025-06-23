---
base_model: distilgpt2
library_name: peft
tags:
- pipeline
- sft_laptop_only
- distilgpt2
- sft
pipeline_tag: text-generation
---

# distilgpt2_quick_test_distilgpt2_sft-laptop-only_example_dataset_distilgpt2_sle5e-05-snu1-spe4-sbfFalse-sfpFalse-sgrFalse-sma512-spaFalse

SFT fine-tuned model trained on the example_dataset dataset.

## Model Details

- **Model Type**: SFT fine-tuned model
- **Base Model**: distilgpt2
- **Dataset**: example_dataset
- **Training Type**: Pipeline
- **Pipeline**: sft_laptop_only

- **Created**: 2025-06-20T17:03:08.531496
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
python -m scripts.inference example_dataset/distilgpt2_quick_test_distilgpt2_sft-laptop-only_example_dataset_distilgpt2_sle5e-05-snu1-spe4-sbfFalse-sfpFalse-sgrFalse-sma512-spaFalse
```

## Files

- `adapter_model.safetensors`: LoRA adapter weights
- `model_config.json`: Model configuration  
- `peft_config.json`: PEFT/LoRA configuration
- `run_metadata.json`: Complete training metadata
- `trainer_state.json`: Detailed training logs
