import torch
from dataclasses import dataclass
from typing import List, Dict
from transformer_lens import HookedTransformer
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.collection.activation_store import ActivationStore


@dataclass
class TransformerLensConfig:
    """Configuration for TransformerLensCollector."""

    model_name: str
    hook_points: List[str]  # e.g. ["blocks.12.hook_resid_post"]
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerLensCollector:
    """Collects activations using TransformerLens."""

    def __init__(self, config: TransformerLensConfig):
        self.config = config
        self.model = HookedTransformer.from_pretrained_no_processing(config.model_name)
        self.model.to(config.device)

    def collect(
        self,
        dataset: TokenizedProbingDataset,
    ) -> Dict[str, ActivationStore]:
        """Collect activations for each hook point.

        Returns:
            Dictionary mapping hook points to ActivationCache objects
        """
        all_activations = {}

        # Set model to evaluation mode
        self.model.eval()

        # Process in batches
        with torch.no_grad():  # Disable gradient computation for determinism
            for batch_start in range(0, len(dataset.examples), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(dataset.examples))
                batch_indices = list(range(batch_start, batch_end))

                # Get batch tensors
                batch = dataset.get_batch_tensors(batch_indices)

                # Run model with caching
                _, cache = self.model.run_with_cache(
                    batch["input_ids"].to(self.config.device),
                    names_filter=self.config.hook_points,
                    return_cache_object=True,
                )

                # Store activations for each hook point
                for hook in self.config.hook_points:
                    if hook not in all_activations:
                        all_activations[hook] = []
                    all_activations[hook].append(cache[hook].cpu())

        # Create ActivationCache objects
        return {
            hook: ActivationStore(
                raw_activations=torch.cat(activations, dim=0),
                hook_point=hook,
                example_indices=torch.arange(len(dataset.examples)),
                sequence_lengths=torch.tensor(dataset.get_token_lengths()),
                hidden_size=activations[0].shape[-1],
                dataset=dataset,
                labels=torch.tensor([ex.label for ex in dataset.examples]),
                label_texts=[ex.label_text for ex in dataset.examples],
            )
            for hook, activations in all_activations.items()
        }
