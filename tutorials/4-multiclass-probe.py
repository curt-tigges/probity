# %% [markdown]
# # Multi-Class Logistic Probe Tutorial
# This tutorial demonstrates how to create, train, and use the `MultiClassLogisticProbe`.
# We will create a simple dataset where the task is to identify the starting letter
# of a word from a small set of categories (e.g., 'A', 'B', 'C', 'Other').

# %% [markdown]
# ## Setup
# Import necessary libraries and set up device/seed.

# %%
import torch
import numpy as np
import random
import torch.backends
import matplotlib.pyplot as plt
from typing import List, Dict, Union

from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.datasets.position_finder import Position, PositionFinder
from probity.probes import MultiClassLogisticProbe, MultiClassLogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.probes.inference import ProbeInference
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

# Set torch device consistently
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")


# Function to set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Set seed
set_seed(42)

# %% [markdown]
# ## 1. Dataset Creation
# We create a dataset of words and classify them based on their starting letter.
# Categories: 0='A', 1='B', 2='C', 3='Other'.
# We will use `PositionFinder` to mark the position of the first character.

# %%
# Define words and categories
words = [
    # A words
    "Apple",
    "Ant",
    "Ace",
    "Art",
    "Actor",
    "Anchor",
    "Arrow",
    "Angel",
    "Autumn",
    "Attic",
    "Apron",
    "Author",
    "Airplane",
    "Argument",
    "Avenue",
    "Award",
    "Action",
    "Adventure",
    "Answer",
    "Article",
    # B words
    "Ball",
    "Bat",
    "Box",
    "Bus",
    "Banana",
    "Bear",
    "Book",
    "Boat",
    "Bread",
    "Brick",
    "Bridge",
    "Button",
    "Bubble",
    "Branch",
    "Breeze",
    "Bottle",
    "Balance",
    "Basket",
    "Beach",
    "Bedroom",
    # C words
    "Cat",
    "Cup",
    "Car",
    "Cow",
    "Cake",
    "Coat",
    "Coin",
    "Clock",
    "Cloud",
    "Castle",
    "Circle",
    "Candle",
    "Camera",
    "Canvas",
    "Corner",
    "Country",
    "Color",
    "Comfort",
    "Coffee",
    "Crayon",
    # Other starting letters
    "Dog",
    "Duck",
    "Door",
    "Desk",
    "Drum",
    "Dream",
    "Dance",
    "Diamond",
    "Elephant",
    "Egg",
    "Ear",
    "Eye",
    "Eagle",
    "Earth",
    "Engine",
    "Elbow",
    "Fish",
    "Fan",
    "Fox",
    "Frog",
    "Fire",
    "Flag",
    "Floor",
    "Flower",
    "Goat",
    "Gum",
    "Gas",
    "Gift",
    "Grape",
    "Glass",
    "Glove",
    "Grass",
    "Hat",
    "Hand",
    "Heart",
    "House",
    "Honey",
    "Horse",
    "Hammer",
    "Harp",
    "Island",
    "Ice",
    "Ink",
    "Idea",
    "Image",
    "Iron",
    "Insect",
    "Igloo",
    "Jacket",
    "Jar",
    "Jelly",
    "Jewel",
    "Joke",
    "Juice",
    "Jungle",
    "Journey",
    "Kite",
    "Key",
    "King",
    "Kitten",
    "Knife",
    "Knee",
    "Koala",
    "Kitchen",
    "Lamp",
    "Leaf",
    "Lion",
    "Lake",
    "Lemon",
    "Letter",
    "Light",
    "Lizard",
    "Map",
    "Moon",
    "Mouse",
    "Milk",
    "Money",
    "Music",
    "Mirror",
    "Mountain",
    "Nail",
    "Nest",
    "Nose",
    "Note",
    "Nurse",
    "Night",
    "Needle",
    "Number",
    "Orange",
    "Owl",
    "Ocean",
    "Oven",
    "Olive",
    "Onion",
    "Office",
    "Orbit",
    "Pen",
    "Pig",
    "Pie",
    "Pot",
    "Paper",
    "Paint",
    "Peach",
    "Pillow",
    "Queen",
    "Quilt",
    "Quiet",
    "Question",
    "Quick",
    "Quiver",
    "Quote",
    "Quiz",
    "Rabbit",
    "Rain",
    "Ring",
    "Road",
    "Robot",
    "River",
    "Rocket",
    "Rope",
    "Sun",
    "Star",
    "Shoe",
    "Ship",
    "Snow",
    "Soup",
    "Spoon",
    "Squirrel",
    "Table",
    "Tree",
    "Tiger",
    "Train",
    "Toast",
    "Thumb",
    "Turtle",
    "Tower",
    "Umbrella",
    "Unicorn",
    "Uniform",
    "Urn",
    "Uncle",
    "Unit",
    "User",
    "Utensil",
    "Vase",
    "Violin",
    "Volcano",
    "Van",
    "Vegetable",
    "Village",
    "Voice",
    "Vulture",
    "Watch",
    "Water",
    "Wheel",
    "Wind",
    "Worm",
    "Whale",
    "Window",
    "Wolf",
    "Xylophone",
    "Xerox",
    "X-ray",  # Limited X words
    "Yarn",
    "Yacht",
    "Yolk",
    "Yogurt",
    "Yellow",
    "Yard",
    "Yawn",
    "Youth",
    "Zebra",
    "Zipper",
    "Zoo",
    "Zero",
    "Zone",
    "Zigzag",  # Limited Z words
]

letter_map = {"A": 0, "B": 1, "C": 2}
num_classes = 4  # A, B, C, Other
label_map_inv = {0: "A", 1: "B", 2: "C", 3: "Other"}

# Position finder for the first character
first_char_finder = PositionFinder.from_char_position(0)

# Create ProbingExamples
start_letter_examples: List[ProbingExample] = []
for word in words:
    first_letter = word[0].upper()
    label = letter_map.get(first_letter, 3)  # Default to 'Other' (label 3)
    label_text = label_map_inv[label]

    # Find the position of the first character
    positions_dict = {}
    pos = first_char_finder(word)
    if pos:
        # PositionFinder returns a single Position object
        positions_dict["FIRST_CHAR"] = pos

    start_letter_examples.append(
        ProbingExample(
            text=word,
            label=label,  # Numeric label (0, 1, 2, 3)
            label_text=label_text,
            character_positions=(
                CharacterPositions(positions_dict) if positions_dict else None
            ),
            attributes={"start_letter": first_letter},
        )
    )

# Create the ProbingDataset
start_letter_dataset = ProbingDataset(
    examples=start_letter_examples,
    task_type="classification",
    label_mapping={
        v: k for k, v in label_map_inv.items()
    },  # Store inverse map for clarity
    dataset_attributes={
        "description": "Starting letter classification (A, B, C, Other)"
    },
)

# Display some examples
print("Starting Letter Dataset Examples:")
for i in np.random.choice(
    range(len(start_letter_dataset.examples)), size=5, replace=False
):
    ex = start_letter_dataset.examples[i]
    print(
        f"  '{ex.text}' -> Label: {ex.label} ({ex.label_text}), Pos: {ex.character_positions['FIRST_CHAR'] if ex.character_positions else 'N/A'}"
    )

print(f"\nPosition types: {start_letter_dataset.position_types}")
print(f"Label mapping: {start_letter_dataset.label_mapping}")

# %% [markdown]
# ## 2. Tokenization
# Tokenize the dataset using a pre-trained tokenizer. We pay attention to padding
# and special tokens. The `TokenizedProbingDataset` handles the conversion of
# character positions to token positions automatically.

# %%
# Set up tokenizer
# Using a small model for faster demonstration
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
# We add special tokens (like BOS) and pad to a fixed length.
tokenized_start_letter_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=start_letter_dataset,
    tokenizer=tokenizer,
    padding="max_length",
    max_length=8,  # Short max length for single words
    add_special_tokens=True,  # Adds BOS/EOS if applicable
)

print(f"\nTokenized Dataset Size: {len(tokenized_start_letter_dataset.examples)}")
print(
    f"Max token length: {max(len(ex.tokens) for ex in tokenized_start_letter_dataset.examples)}"
)

# Verify token positions for a sample
sample_idx = 0
sample_tokenized_ex = tokenized_start_letter_dataset.examples[sample_idx]
print(f"\nSample Example {sample_idx}:")
print(f"  Text: '{sample_tokenized_ex.text}'")
print(f"  Tokens: {sample_tokenized_ex.tokens}")
print(
    f"  Decoded Tokens: {[tokenizer.decode([t]) for t in sample_tokenized_ex.tokens]}"
)
if sample_tokenized_ex.token_positions:
    first_char_token_pos = sample_tokenized_ex.token_positions["FIRST_CHAR"]
    print(f"  Token Position for FIRST_CHAR: {first_char_token_pos}")
    if 0 <= first_char_token_pos < len(sample_tokenized_ex.tokens):
        target_token_id = sample_tokenized_ex.tokens[first_char_token_pos]
        print(
            f"  Token at target position: {target_token_id} ('{tokenizer.decode([target_token_id])}')"
        )
    else:
        print("  Target token position is out of bounds!")
else:
    print("  Token positions not found for sample.")

# You can use verify_position_tokens for more detailed checking:
# verification_results = tokenized_start_letter_dataset.verify_position_tokens(tokenizer)
# print("\nVerification Results (first few):")
# for i, res in verification_results.items():
#     if i < 3: print(f" Example {i}: {res}")

# %% [markdown]
# ## 3. Probe Training
# Train the `MultiClassLogisticProbe` using the `SupervisedProbeTrainer`.
# We need to configure the probe with the correct input size (model's hidden dimension)
# and output size (number of classes).

# %%
# Configure model and hook point
# Using a mid-layer residual stream hook point
hook_point = "blocks.1.hook_resid_pre"
layer = 1

# Get model's hidden dimension
model = HookedTransformer.from_pretrained(model_name, device=device)
hidden_size = model.cfg.d_model
print(f"\nModel {model_name} hidden dimension: {hidden_size}")

# Configure the MultiClassLogisticProbe
probe_config = MultiClassLogisticProbeConfig(
    input_size=hidden_size,
    output_size=num_classes,  # 4 classes: A, B, C, Other
    normalize_weights=True,
    bias=True,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=layer,
    name="start_letter_multiclass_probe",
)

# Configure the Trainer
# Note: Handle class imbalance is useful here as 'Other' is more frequent
# Ensure standardize_activations is False as Probe expects raw data for inference
trainer_config = SupervisedTrainerConfig(
    batch_size=8,
    learning_rate=5e-3,  # Slightly higher LR can work for simple tasks
    num_epochs=20,  # More epochs for convergence
    weight_decay=0.01,
    train_ratio=0.7,
    handle_class_imbalance=True,  # Important for multi-class
    show_progress=True,
    device=device,
    patience=5,  # Add early stopping
    min_delta=1e-4,
    standardize_activations=False,  # Important: Keep False if probe needs raw activations
)

# Revert to using ProbePipeline
pipeline_config = ProbePipelineConfig(
    dataset=tokenized_start_letter_dataset,
    probe_cls=MultiClassLogisticProbe,
    probe_config=probe_config,
    trainer_cls=SupervisedProbeTrainer,
    trainer_config=trainer_config,
    position_key="FIRST_CHAR",
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/start_letter_probe_cache",  # Specify cache directory
    device=device,  # Specify device
)

# Train the probe using the pipeline's run method
print(f"\nTraining Multi-Class Probe for layer {layer} using Pipeline...")
pipeline: ProbePipeline = ProbePipeline(pipeline_config)

# Run the pipeline - this handles activation collection, training, etc.
start_letter_probe, history = pipeline.run(hook_point=hook_point)

print("Training complete.")

# %% [markdown]
# ## 4. Plot Training Results

# %%
# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss (Cross Entropy)")
plt.title(f"Multi-Class Probe Training History (Layer {layer})")
plt.legend()
plt.grid(True)
plt.show()

print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")

# %% [markdown]
# ## 5. Inference
# Use the trained multi-class probe with `ProbeInference` to predict the starting
# letter category for new words. We check the probability distribution output by
# the probe at the first token's position.

# %%
# Create ProbeInference instance
start_letter_inference = ProbeInference(
    model_name=model_name,
    hook_point=hook_point,
    probe=start_letter_probe,
    device=device,
)

# Test words
test_words = [
    "Antelope",
    "Bear",
    "Crocodile",
    "Zebra",
    "Banana",
    "Avocado",
    "Gorilla",
    "Camel",
]

# Get probabilities (softmax applied for MultiClassLogisticProbe)
# Shape: (batch_size, seq_len, num_classes)
probabilities = start_letter_inference.get_probabilities(test_words)

print(f"\n===== Inference Results (Layer {layer}) =====")
print(f"Probing at the first character's token position.")
print(f"Classes: {label_map_inv}")

# Ensure probe is in eval mode for inference
start_letter_probe.eval()
for i, word in enumerate(test_words):
    print(f"\nWord: '{word}'")

    # Tokenize the single word to find the token position corresponding to the first char
    # Note: Need consistent tokenization settings with training
    tokens_info = tokenizer(word, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens_info["input_ids"][0]
    decoded_tokens = [tokenizer.decode([t]) for t in input_ids]

    # Find token position for the first character (index 0)
    # This uses the tokenizer directly, similar to how TokenizedProbingDataset does it.
    # We assume the first character corresponds to the first non-special token.
    first_char_token_index = -1
    for idx, token_id in enumerate(input_ids.tolist()):
        if token_id not in tokenizer.all_special_ids:
            first_char_token_index = idx
            break

    if first_char_token_index == -1 and len(input_ids) > 0:
        # Fallback if only special tokens? Use first token.
        first_char_token_index = 0

    print(f"  Tokens: {input_ids.tolist()}")
    print(f"  Decoded: {decoded_tokens}")
    print(f"  Inferred First Char Token Index: {first_char_token_index}")

    if first_char_token_index != -1 and first_char_token_index < probabilities.shape[1]:
        # Get the probability distribution at the first character's token position
        # Probabilities shape: [batch, seq_len, num_classes]
        prob_dist = probabilities[i, first_char_token_index, :]
        predicted_class_index = torch.argmax(prob_dist).item()
        predicted_class_label = label_map_inv[predicted_class_index]
        confidence = prob_dist[predicted_class_index].item()

        print(
            f"  Probabilities at token {first_char_token_index} ('{tokenizer.decode([input_ids[first_char_token_index]])}'):"
        )
        for cls_idx, prob in enumerate(prob_dist):
            print(f"    Class {cls_idx} ({label_map_inv[cls_idx]}): {prob:.4f}")
        print(
            f"  Prediction: Starts with '{predicted_class_label}' (Confidence: {confidence:.4f})"
        )
    else:
        print("  Could not determine valid token index for the first character.")

# %% [markdown]
# ## Conclusion
# This tutorial showed how to set up and train a `MultiClassLogisticProbe`.
# Key steps included:
# - Defining a multi-class dataset with integer labels.
# - Using `MultiClassLogisticProbeConfig` with `output_size` > 1.
# - Training with `SupervisedProbeTrainer`, which automatically handles `CrossEntropyLoss` and class imbalance for this probe type.
# - Performing inference using `ProbeInference`, where `get_probabilities` applies `softmax` to the multi-class probe outputs.
#
# This probe type is useful for analyzing how models represent categorical features with more than two options within their activations.
# %%
