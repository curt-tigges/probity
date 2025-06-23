# %% [markdown]
# # Dataset Creation Tutorial
# This file demonstrates several ways to create datasets in Probity:
# 1. Creating a custom templated dataset (question classification)
# 2. Creating a non-templated dataset from scratch (facts vs. opinions)
# 3. Tokenizing both datasets
# 4. Training probes on each dataset
# 5. Comparing the results

# %% [markdown]
# ## Setup

# %%
import torch
import numpy as np
import random
import torch.backends
import matplotlib.pyplot as plt

from probity.datasets.templated import TemplatedDataset, TemplateVariable, Template
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions
from probity.datasets.position_finder import Position, PositionFinder
from transformers import AutoTokenizer
from probity.probes import LogisticProbe, LogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.probes.inference import ProbeInference
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
from transformer_lens import HookedTransformer

# Set torch device consistently
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Set seed for reproducibility
set_seed(42)

# %% [markdown]
# ## Dataset Creation Method 1: Custom Templated Dataset
# First, we'll create a custom templated dataset for question classification.
# We'll distinguish between yes/no questions and open-ended questions.

# %%
# Create question classification dataset using templates
question_words = {
    "yes_no": ["Do", "Did", "Will", "Would", "Can"],
    "open_ended": ["What", "Why", "How"],
}

subjects = [
    "scientists",
    "doctors",
    "students",
    "researchers",
    "experts",
    "philosophers",
    "historians",
    "engineers",
]

verbs = [
    "think",
    "believe",
    "understand",
    "know",
    "observe",
    "theorize",
    "hypothesize",
    "discover",
]

objects = [
    "this concept",
    "these methods",
    "the solution",
    "the problem",
    "the results",
    "the experiment",
    "the evidence",
    "the analysis",
]

# Create TemplateVariables
question_word_var = TemplateVariable(
    name="Q_WORD",
    values=[word for lst in question_words.values() for word in lst],
    attributes={"question_type": [k for k, v in question_words.items() for _ in v]},
    class_bound=True,
    class_key="question_type",
)

subject_var = TemplateVariable(name="SUBJECT", values=subjects)

verb_var = TemplateVariable(name="VERB", values=verbs)

object_var = TemplateVariable(name="OBJECT", values=objects)

# Create templates
template1 = Template(
    template="{Q_WORD} {SUBJECT} {VERB} {OBJECT}?",
    variables={
        "Q_WORD": question_word_var,
        "SUBJECT": subject_var,
        "VERB": verb_var,
        "OBJECT": object_var,
    },
    attributes={"task": "question_classification"},
)

# Create the dataset
question_dataset = TemplatedDataset(
    templates=[template1], attributes={"description": "Question classification dataset"}
)

# Convert to probing dataset
question_probing_dataset = question_dataset.to_probing_dataset(
    label_from_attributes="question_type",
    label_map={"yes_no": 1, "open_ended": 0},
    auto_add_positions=True,
)

# Add position finder for the question mark instead of the question word
question_mark_finder = PositionFinder.from_regex(r"\?")
question_probing_dataset.add_target_positions(
    key="QUESTION_MARK", finder=question_mark_finder
)


# %% [markdown]
# ### Dataset 1 Results
# Below we see some random examples from the dataset, as well as the position types and some statistics.
# These sentences are relatively short and trivial, but serve well to demonstrate the concept.

# %%
# Display dataset statistics and examples
print(f"Dataset size: {len(question_probing_dataset.examples)}")
print(
    f"Max sequence length: {max(len(ex.text) for ex in question_probing_dataset.examples)}"
)

print("\nQuestion Classification Dataset Examples:")
for i in np.random.choice(
    range(len(question_probing_dataset.examples)), size=6, replace=False
):
    ex = question_probing_dataset.examples[i]
    label = "yes/no question" if ex.label == 1 else "open-ended question"
    print(f"Example {i}: '{ex.text}' (Label: {label})")

# Verify positions
sample_ex = question_probing_dataset.examples[0]
print(f"\nPosition types: {question_probing_dataset.position_types}")
if question_probing_dataset.position_types:
    for key in question_probing_dataset.position_types:
        if (
            sample_ex.character_positions
            and key in sample_ex.character_positions.keys()
        ):
            pos = sample_ex.character_positions[key]
            if isinstance(pos, Position):
                print(
                    f"{key} position: {pos.start}-{pos.end} "
                    f"('{sample_ex.text[pos.start:pos.end]}')"
                )
            else:
                for i, p in enumerate(pos):
                    print(
                        f"{key} position {i}: {p.start}-{p.end} "
                        f"('{sample_ex.text[p.start:p.end]}')"
                    )


# %% [markdown]
# ## Dataset Creation Method 2: Code vs. Not-Code Dataset
# Now we'll create a dataset for distinguishing between code and non-code text.
# A probe on this dataset should learn a context feature for the presence of code.

# %%
# Create code vs. not-code dataset
code_examples = [
    "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
    "for i in range(10):\n    print(i**2)",
    "import numpy as np\nnp.array([1, 2, 3])",
    "class Person:\n    def __init__(self, name):\n        self.name = name",
    "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    print('Error')",
    "with open('file.txt', 'r') as f:\n    data = f.read()",
    "lambda x: x * 2",
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
    "async def fetch_data():\n    return await api.get_data()",
    "@decorator\ndef my_function():\n    pass",
    "x = {'a': 1, 'b': 2}",
    "if x > 0 and y < 10:\n    print('Valid')",
    "while not done:\n    process()",
    "result = [i*i for i in range(10)]",
    "def map_function(f, items):\n    return [f(x) for x in items]",
]

non_code_examples = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "It was the best of times, it was the worst of times.",
    "In a hole in the ground there lived a hobbit.",
    "The only thing we have to fear is fear itself.",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation.",
    "I have a dream that one day this nation will rise up.",
    "Ask not what your country can do for you, ask what you can do for your country.",
    "That's one small step for man, one giant leap for mankind.",
    "Life is like a box of chocolates. You never know what you're gonna get.",
    "May the Force be with you.",
    "Elementary, my dear Watson.",
    "The greatest glory in living lies not in never falling, but in rising every time we fall.",
    "The way to get started is to quit talking and begin doing.",
    "You miss 100% of the shots you don't take.",
]

# Create position finder for the end of each example
# PositionFinder can take a variety of inputs, including regexes, character positions, and templates
end_finder = PositionFinder.from_regex(r"[\S].{0,1}$")

# Create ProbingExamples
code_vs_text_examples = []

# Add code examples
for text in code_examples:
    positions_dict = {}
    end_pos = end_finder(text)
    if end_pos:
        positions_dict["END_POSITION"] = end_pos[0]

    code_vs_text_examples.append(
        ProbingExample(
            text=text,
            label=1,  # 1 for code
            label_text="code",
            character_positions=(
                CharacterPositions(positions_dict) if positions_dict else None
            ),
            attributes={"type": "code"},
        )
    )

# Add non-code examples
for text in non_code_examples:
    positions_dict = {}
    end_pos = end_finder(text)
    if end_pos:
        positions_dict["END_POSITION"] = end_pos[0]

    code_vs_text_examples.append(
        ProbingExample(
            text=text,
            label=0,  # 0 for non-code
            label_text="non_code",
            character_positions=(
                CharacterPositions(positions_dict) if positions_dict else None
            ),
            attributes={"type": "non_code"},
        )
    )

# Create the dataset
code_dataset = ProbingDataset(
    examples=code_vs_text_examples,
    task_type="classification",
    label_mapping={"non_code": 0, "code": 1},
    dataset_attributes={"description": "Code vs. non-code classification dataset"},
)

# %% [markdown]
# ### Dataset 2 Results
# Once again, we can display some examples, as well as the position types and some statistics.

# Display examples from the code vs. non-code dataset
print("Code vs. Non-Code Dataset Examples:")
for i in np.random.choice(range(len(code_dataset.examples)), size=6, replace=False):
    ex = code_dataset.examples[i]
    label = "code" if ex.label == 1 else "non-code"
    print(f"Example {i}: '{ex.text[:50]}...' (Label: {label})")

# Verify positions
sample_ex = code_dataset.examples[0]
print(f"\nPosition types: {code_dataset.position_types}")
if code_dataset.position_types:
    for key in code_dataset.position_types:
        if (
            sample_ex.character_positions
            and key in sample_ex.character_positions.keys()
        ):
            pos = sample_ex.character_positions[key]
            if isinstance(pos, Position):
                print(f"{key} position: {pos.start}-{pos.end}")
            else:
                for i, p in enumerate(pos):
                    print(f"{key} position {i}: {p.start}-{p.end}")

# %% [markdown]
# ## Dataset 3: First-Person vs Third-Person Perspective
# Now we'll create a dataset for distinguishing between first-person and third-person perspectives.
# This method is the same as Dataset 2, but with a different approach to finding the target position.

# %%
# Create dataset for first-person vs third-person perspective
first_person_statements = [
    "I think artificial intelligence is fascinating.",
    "I feel excited about the future of technology.",
    "I believe climate change requires immediate action.",
    "I want to learn more about quantum computing.",
    "I remember my first day at college vividly.",
    "I prefer reading books to watching movies.",
    "I hope to travel to Japan someday.",
    "I enjoy solving complex mathematical problems.",
    "I consider philosophy to be essential for critical thinking.",
    "I dream of starting my own company.",
    "I wish people would be kinder to each other.",
    "I need more time to complete this project.",
    "I wonder what the world will be like in 100 years.",
    "I regret not learning to play an instrument.",
    "I understand the importance of regular exercise.",
    "I desire to improve my public speaking skills.",
    "I plan to learn a new language next year.",
    "I question many assumptions about human nature.",
    "I value honesty above all other qualities.",
    "I imagine a world without poverty or hunger.",
]

third_person_statements = [
    "He thinks artificial intelligence is fascinating.",
    "She feels excited about the future of technology.",
    "They believe climate change requires immediate action.",
    "He wants to learn more about quantum computing.",
    "She remembers her first day at college vividly.",
    "He prefers reading books to watching movies.",
    "She hopes to travel to Japan someday.",
    "They enjoy solving complex mathematical problems.",
    "She considers philosophy to be essential for critical thinking.",
    "He dreams of starting his own company.",
    "She wishes people would be kinder to each other.",
    "He needs more time to complete this project.",
    "They wonder what the world will be like in 100 years.",
    "She regrets not learning to play an instrument.",
    "He understands the importance of regular exercise.",
    "She desires to improve her public speaking skills.",
    "They plan to learn a new language next year.",
    "He questions many assumptions about human nature.",
    "She values honesty above all other qualities.",
    "They imagine a world without poverty or hunger.",
]

# Create position finder for the beginning pronoun (I, He, She, They)
pronoun_finder = PositionFinder.from_regex(r"^(I|He|She|They)\b")

# Create ProbingExamples
perspective_examples = []

# Add first-person examples
for text in first_person_statements:
    positions_dict = {}
    pronoun_pos = pronoun_finder(text)
    if pronoun_pos:
        positions_dict["PRONOUN_POSITION"] = pronoun_pos[0]

    perspective_examples.append(
        ProbingExample(
            text=text,
            label=1,  # 1 for first-person
            label_text="first_person",
            character_positions=(
                CharacterPositions(positions_dict) if positions_dict else None
            ),
            attributes={"perspective": "first_person"},
        )
    )

# Add third-person examples
for text in third_person_statements:
    positions_dict = {}
    pronoun_pos = pronoun_finder(text)
    if pronoun_pos:
        positions_dict["PRONOUN_POSITION"] = pronoun_pos[0]

    perspective_examples.append(
        ProbingExample(
            text=text,
            label=0,  # 0 for third-person
            label_text="third_person",
            character_positions=(
                CharacterPositions(positions_dict) if positions_dict else None
            ),
            attributes={"perspective": "third_person"},
        )
    )

# Create the dataset
perspective_dataset = ProbingDataset(
    examples=perspective_examples,
    task_type="classification",
    label_mapping={"third_person": 0, "first_person": 1},
    dataset_attributes={
        "description": "First-person vs third-person perspective classification dataset"
    },
)

# %%

# Display examples from the perspective dataset
print("First-Person vs Third-Person Dataset Examples:")
for i in np.random.choice(
    range(len(perspective_dataset.examples)), size=6, replace=False
):
    ex = perspective_dataset.examples[i]
    label = "first-person" if ex.label == 1 else "third-person"
    print(f"Example {i}: '{ex.text}' (Label: {label})")

# Verify positions
sample_ex = perspective_dataset.examples[0]
print(f"\nPosition types: {perspective_dataset.position_types}")
if perspective_dataset.position_types:
    for key in perspective_dataset.position_types:
        if (
            sample_ex.character_positions
            and key in sample_ex.character_positions.keys()
        ):
            pos = sample_ex.character_positions[key]
            if isinstance(pos, Position):
                print(
                    f"{key} position: {pos.start}-{pos.end} "
                    f"('{sample_ex.text[pos.start:pos.end]}')"
                )
            else:
                for i, p in enumerate(pos):
                    print(
                        f"{key} position {i}: {p.start}-{p.end} "
                        f"('{sample_ex.text[p.start:p.end]}')"
                    )

# %% [markdown]
# ## Tokenization
# Now that we have our datasets, we need to tokenize them for use with our model.

# %%
# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the question dataset
tokenized_question_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=question_probing_dataset,
    tokenizer=tokenizer,
    padding="max_length",
    max_length=12,
    add_special_tokens=True,
)
print("\nQuestion Dataset:")
print(f"Dataset size: {len(tokenized_question_dataset.examples)}")
print(
    f"Max token length: {max(len(ex.tokens) for ex in tokenized_question_dataset.examples)}"
)

# Tokenize the code dataset
tokenized_code_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=code_dataset,
    tokenizer=tokenizer,
    padding="max_length",
    max_length=128,  # Longer max_length for code examples
    add_special_tokens=True,
)
print("\nCode Dataset:")
print(f"Dataset size: {len(tokenized_code_dataset.examples)}")
print(
    f"Max token length: {max(len(ex.tokens) for ex in tokenized_code_dataset.examples)}"
)

# Tokenize the perspective dataset
tokenized_perspective_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=perspective_dataset,
    tokenizer=tokenizer,
    padding="max_length",
    max_length=24,
    add_special_tokens=True,
)
print("\nPerspective Dataset:")
print(f"Dataset size: {len(tokenized_perspective_dataset.examples)}")
print(
    f"Max token length: {max(len(ex.tokens) for ex in tokenized_perspective_dataset.examples)}"
)

# %% [markdown]
# ## Probe Training
# Now let's train probes for each dataset to see if we can detect their respective features.
# We'll try multiple model layers to see which ones work best.

# %% Configure models and hook points
model_name = "gpt2"

# IMPROVEMENT: Test multiple hook points (layers) to find the best one
hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in [4, 8, 11]]

# Get model's hidden dimension dynamically
model = HookedTransformer.from_pretrained(model_name, device=device)
hidden_size = model.cfg.d_model
print(f"Model {model_name} has hidden dimension: {hidden_size}")

# Common trainer configuration
trainer_config = SupervisedTrainerConfig(
    batch_size=16,
    learning_rate=1e-3,
    num_epochs=10,
    weight_decay=0.01,
    train_ratio=0.7,  # 70-30 train-val split
    handle_class_imbalance=True,
    show_progress=True,
    device=device,
)

# %% Train question type probes across multiple layers
question_probes = {}
question_training_histories = {}

for hook_point in hook_points:
    layer = int(hook_point.split(".")[1])

    # Set up probe configuration
    question_probe_config = LogisticProbeConfig(
        input_size=hidden_size,  # Use dynamically determined hidden size
        normalize_weights=True,
        bias=True,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=layer,
        name=f"question_type_probe_layer_{layer}",
    )

    # Create pipeline config
    question_pipeline_config = ProbePipelineConfig(
        dataset=tokenized_question_dataset,
        probe_cls=LogisticProbe,
        probe_config=question_probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="QUESTION_MARK",  # Changed from Q_WORD to QUESTION_MARK
        model_name=model_name,
        hook_points=[hook_point],
        cache_dir=f"./cache/question_probe_cache_layer_{layer}",
    )

    # Train the probe
    print(f"Training Question Type Probe for layer {layer}...")
    question_pipeline: ProbePipeline = ProbePipeline(question_pipeline_config)
    probe, history = question_pipeline.run()

    # Store results
    question_probes[layer] = probe
    question_training_histories[layer] = history


# %% Train code detection probes across multiple layers
code_probes = {}
code_training_histories = {}

for hook_point in hook_points:
    layer = int(hook_point.split(".")[1])

    # Set up probe configuration
    code_probe_config = LogisticProbeConfig(
        input_size=hidden_size,  # Use dynamically determined hidden size
        normalize_weights=True,
        bias=True,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=layer,
        name=f"code_probe_layer_{layer}",
    )

    # Create pipeline config
    code_pipeline_config = ProbePipelineConfig(
        dataset=tokenized_code_dataset,
        probe_cls=LogisticProbe,
        probe_config=code_probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="END_POSITION",  # Probe at the end of the example
        model_name=model_name,
        hook_points=[hook_point],
        cache_dir=f"./cache/code_probe_cache_layer_{layer}",
    )

    # Train the probe
    print(f"Training Code Detection Probe for layer {layer}...")
    code_pipeline: ProbePipeline = ProbePipeline(code_pipeline_config)
    probe, history = code_pipeline.run()

    # Store results
    code_probes[layer] = probe
    code_training_histories[layer] = history

# %% Train perspective probes across multiple layers
perspective_probes = {}
perspective_training_histories = {}

for hook_point in hook_points:
    layer = int(hook_point.split(".")[1])

    # Set up probe configuration
    perspective_probe_config = LogisticProbeConfig(
        input_size=hidden_size,  # Use dynamically determined hidden size
        normalize_weights=True,
        bias=True,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=layer,
        name=f"perspective_probe_layer_{layer}",
    )

    # Create pipeline config
    perspective_pipeline_config = ProbePipelineConfig(
        dataset=tokenized_perspective_dataset,
        probe_cls=LogisticProbe,
        probe_config=perspective_probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="PRONOUN_POSITION",  # Probe at the pronoun position
        model_name=model_name,
        hook_points=[hook_point],
        cache_dir=f"./cache/perspective_probe_cache_layer_{layer}",
    )

    # Train the probe
    print(f"Training Perspective Probe for layer {layer}...")
    perspective_pipeline: ProbePipeline = ProbePipeline(perspective_pipeline_config)
    probe, history = perspective_pipeline.run()

    # Store results
    perspective_probes[layer] = probe
    perspective_training_histories[layer] = history

# %% Plot training histories across layers
plt.figure(figsize=(15, 20))

# Plot question probe training histories
plt.subplot(3, 1, 1)
for layer, history in question_training_histories.items():
    plt.plot(history["train_loss"], label=f"Layer {layer} Train")
    plt.plot(history["val_loss"], label=f"Layer {layer} Val", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Question Type Probe Training History Across Layers")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot code probe training histories
plt.subplot(3, 1, 2)
for layer, history in code_training_histories.items():
    plt.plot(history["train_loss"], label=f"Layer {layer} Train")
    plt.plot(history["val_loss"], label=f"Layer {layer} Val", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Code Detection Probe Training History Across Layers")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot perspective probe training histories
plt.subplot(3, 1, 3)
for layer, history in perspective_training_histories.items():
    plt.plot(history["train_loss"], label=f"Layer {layer} Train")
    plt.plot(history["val_loss"], label=f"Layer {layer} Val", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Perspective Probe Training History Across Layers")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()

# %% Compare layer performance and find the best layer for each task
layer_results = {}

for task, histories in [
    ("question_type", question_training_histories),
    ("code_detection", code_training_histories),
    ("perspective", perspective_training_histories),
]:
    layer_results[task] = {}
    for layer, history in histories.items():
        # Use the final validation loss as the metric
        final_val_loss = history["val_loss"][-1]
        layer_results[task][layer] = final_val_loss

# Find the best layer for each task
best_question_layer = min(
    layer_results["question_type"], key=lambda k: layer_results["question_type"][k]
)
best_code_layer = min(
    layer_results["code_detection"], key=lambda k: layer_results["code_detection"][k]
)
best_perspective_layer = min(
    layer_results["perspective"], key=lambda k: layer_results["perspective"][k]
)

print(f"Best layer for question type classification: {best_question_layer}")
print(f"Best layer for code detection: {best_code_layer}")
print(f"Best layer for perspective classification: {best_perspective_layer}")

# %% [markdown]
# ## Inference with Best Trained Probes
# Now let's test our best trained probes on some new examples. Note that we've set some manual thresholds for the confidence scores to make the results more prominent.
# However, in practice, you may want to use a more sophisticated approach to setting these thresholds (e.g. a ROC curve). Nevertheless, if you look at the results it should
# be clear that there is a dichotomy between the classes.

# You could also get the "activation in the probe direction" instead of the probability, and then use that to make predictions. This can be done with the
# `ProbeInference.get_direction_activations()` method.

# %%
# Inference with best Question Type Probe
best_hook_point = f"blocks.{best_question_layer}.hook_resid_pre"
question_inference = ProbeInference(
    model_name=model_name,
    hook_point=best_hook_point,
    probe=question_probes[best_question_layer],
    device=device,
)

# Create some test examples
test_questions = [
    "Is artificial intelligence a threat to humanity?",
    "Are electric vehicles better for the environment?",
    "What factors contribute to climate change?",
    "How do quantum computers work?",
    "Will humans colonize Mars within this century?",
    "Where should I go for my next vacation?",
]

# Get probabilities (applies sigmoid for logistic probes)
question_probs = question_inference.get_probabilities(test_questions)

# Display the results
print(f"===== Question Type Probe Results (Layer {best_question_layer}) =====")
print("Probing at the question mark position")
for i, example in enumerate(test_questions):
    print(f"\nText: {example}")
    # Get mean probability across all tokens as an overall score
    overall_score = question_probs[i].mean().item()
    prediction = "Yes/No Question" if overall_score > 0.90 else "Open-Ended Question"
    confidence = max(overall_score, 1 - overall_score)
    print(
        f"Prediction: {prediction} (confidence: {confidence:.4f}) (overall score: {overall_score:.4f})"
    )

# %%
# Inference with best Code Detection Probe
best_hook_point = f"blocks.{best_code_layer}.hook_resid_pre"
code_inference = ProbeInference(
    model_name=model_name,
    hook_point=best_hook_point,
    probe=code_probes[best_code_layer],
    device=device,
)

# Create some test examples
test_code_examples = [
    "from datetime import datetime\nprint(datetime.now())",
    "All the world's a stage, and all the men and women merely players.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "The early bird catches the worm, but the second mouse gets the cheese.",
    "x = 5\ny = 10\nprint(x + y)",
    "Four score and seven years ago our fathers brought forth on this continent a new nation.",
]

# Get probabilities (applies sigmoid for logistic probes)
code_probs = code_inference.get_probabilities(test_code_examples)

# Display the results
print(f"===== Code Detection Probe Results (Layer {best_code_layer}) =====")
print("Probing at the end of examples")
for i, example in enumerate(test_code_examples):
    print(f"\nText: {example[:50]}...")
    # Get mean probability across all tokens as an overall score
    overall_score = code_probs[i].mean().item()
    prediction = "Code" if overall_score > 0.5 else "Not Code"
    confidence = max(overall_score, 1 - overall_score)
    print(
        f"Prediction: {prediction} (confidence: {confidence:.4f}) (overall score: {overall_score:.4f})"
    )

# %%
# Inference with best Perspective Probe
best_hook_point = f"blocks.{best_perspective_layer}.hook_resid_pre"
perspective_inference = ProbeInference(
    model_name=model_name,
    hook_point=best_hook_point,
    probe=perspective_probes[best_perspective_layer],
    device=device,
)

# Create some test examples
test_perspective_statements = [
    "I am going to study computer science in college.",
    "She is going to study computer science in college.",
    "I don't think this approach will work in practice.",
    "They don't think this approach will work in practice.",
    "I always wanted to visit the Grand Canyon.",
    "He always wanted to visit the Grand Canyon.",
]

# Get probabilities (applies sigmoid for logistic probes)
perspective_probs = perspective_inference.get_probabilities(test_perspective_statements)

# Display the results
print(f"===== Perspective Probe Results (Layer {best_perspective_layer}) =====")
print("Probing at the pronoun position")
for i, example in enumerate(test_perspective_statements):
    print(f"\nText: {example}")
    # Get mean probability across all tokens as an overall score
    overall_score = perspective_probs[i].mean().item()
    prediction = "First-Person" if overall_score < 0.70 else "Third-Person"
    confidence = max(overall_score, 1 - overall_score)
    print(
        f"Prediction: {prediction} (confidence: {confidence:.4f}) (overall score: {overall_score:.4f})"
    )

# %% [markdown]
# ## Conclusion
#
# This tutorial demonstrated multiple approaches to creating datasets with Probity:
#
# 1. Using the **TemplatedDataset** class to create structured datasets with predefined
#    patterns and variables - demonstrated with our question type classification dataset.
#
# 2. Creating **non-templated datasets** directly using ProbingExample objects, as
#    shown with our code detection and perspective classification datasets.
#
# In all cases, we ensured that position information was carefully tracked, allowing our
# probes to focus on specific parts of the input that are most relevant to the
# classification task:
#
# - For question classification, we probed at the question mark position (demonstrating that models often deposit sentence-level information in punctuation at the end of a sentence or phrase)
# - For code detection, we probed at the end of each example
# - For perspective classification, we probed at the initial pronoun position
#
# The results show that:
#
# - Different tasks are best probed from different layers
# - Models encode rich semantic information that can be extracted with linear probes
# - Position-specific probing often gives better insights than whole-sequence probing
#
# For your own projects, consider which dataset creation approach best suits your needs
# based on the structure and source of your data, and carefully select positions that
# are most relevant to your probing task.
# %%
