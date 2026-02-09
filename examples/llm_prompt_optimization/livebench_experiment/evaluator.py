"""
Evaluator for LiveBench Instruction Following prompt optimization.

Uses OFFICIAL LiveBench evaluation scripts for accurate scoring.
Supports:
- IFEval format (original Google constraints)
- IFBench format (new, harder constraints)
- Train/Validation/Test split with random sampling
- Adaptive sample size that increases with evolution progress
- Cascade evaluation (Stage 1 quick filter, Stage 2 comprehensive)
"""

import re
import traceback
import yaml
import os
import sys
import time
import random
import json
import pickle
from typing import Dict, Any, Tuple, List, Optional
from openai import OpenAI
from tqdm import tqdm

# ========================================================================================
# SETUP: Add LiveBench to path
# ========================================================================================
LIVEBENCH_PATH = os.path.join(os.path.dirname(__file__), "LiveBench", "livebench")
if LIVEBENCH_PATH not in sys.path:
    sys.path.insert(0, LIVEBENCH_PATH)

# Import LiveBench evaluation modules
try:
    from if_runner.instruction_following_eval import evaluation_main as ifeval_main
    from if_runner.ifbench import evaluation_lib as ifbench_lib
    from process_results.instruction_following.utils import score_results
    LIVEBENCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import LiveBench modules: {e}")
    print("Make sure LiveBench is installed: cd LiveBench && pip install -e .")
    LIVEBENCH_AVAILABLE = False

# ========================================================================================
# GLOBAL VARIABLES
# ========================================================================================
_DATASET_SPLITS = {}  # Cache for train/validation/test splits (in-process)
_MAX_ITERATIONS = 100  # Maximum iterations (loaded from config)

# Cache directory for cross-process caching
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".evaluation_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

EVALUATION_COUNTER_FILE = os.path.join(CACHE_DIR, "evaluation_counter.json")
DATASET_SPLITS_CACHE_FILE = os.path.join(CACHE_DIR, "dataset_splits_cache.pkl")

# ========================================================================================
# CONFIGURATION LOADING
# ========================================================================================
# Read config.yaml to get model settings
config_file = "config.yaml"
config_path = os.path.join(os.path.dirname(__file__), config_file)

try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Warning: Config file not found at {config_path}, using defaults")
    config = {}

# Get LLM settings from config
llm_config = config.get("llm", {})
api_base = llm_config.get("api_base", "http://localhost:1234/v1")

# Handle both single model and model list configurations
models = llm_config.get("models", [])
if models:
    TASK_MODEL_NAME = models[0].get("name", "default-model")
else:
    TASK_MODEL_NAME = llm_config.get("primary_model", "default-model")

# Get evaluator settings
evaluator_config = config.get("evaluator", {})
MAX_RETRIES = evaluator_config.get("max_retries", 3)
MAX_TOKENS = llm_config.get("max_tokens", 4096)

# Initialize OpenAI client
test_model = OpenAI(base_url=api_base)

# Dataset config path
DATASET_CONFIG_PATH = None


def init_dataset_config_path():
    """Initialize DATASET_CONFIG_PATH based on OPENEVOLVE_PROMPT."""
    global DATASET_CONFIG_PATH
    
    if DATASET_CONFIG_PATH is not None:
        return
    
    prompt_file = os.environ.get("OPENEVOLVE_PROMPT")
    if not prompt_file:
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        DATASET_CONFIG_PATH = os.path.join(evaluator_dir, "livebench_prompt_dataset.yaml")
    else:
        basename = os.path.basename(prompt_file)
        dataset_filename = basename.replace("_prompt.txt", "_prompt_dataset.yaml").replace(
            ".txt", "_dataset.yaml"
        )
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        DATASET_CONFIG_PATH = os.path.join(evaluator_dir, dataset_filename)


init_dataset_config_path()


# ========================================================================================
# CROSS-PROCESS CACHING FUNCTIONS
# ========================================================================================

def get_current_iteration_from_checkpoint():
    """
    Get current iteration number from the most recent checkpoint.
    
    Returns:
        int: Current iteration (0 if new run or checkpoint not found)
    """
    try:
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(evaluator_dir, "openevolve_output", "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            return 0
        
        latest_checkpoint_path = None
        latest_timestamp = 0
        latest_iteration = 0
        
        for item in os.listdir(checkpoints_dir):
            checkpoint_path = os.path.join(checkpoints_dir, item)
            if not os.path.isdir(checkpoint_path) or not item.startswith("checkpoint_"):
                continue
            
            metadata_path = os.path.join(checkpoint_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            
            try:
                checkpoint_time = os.path.getmtime(metadata_path)
                
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    checkpoint_iteration = metadata.get("last_iteration", 0)
                    if checkpoint_iteration == 0:
                        try:
                            checkpoint_iteration = int(item.replace("checkpoint_", ""))
                        except ValueError:
                            continue
                
                if checkpoint_time > latest_timestamp:
                    latest_timestamp = checkpoint_time
                    latest_checkpoint_path = checkpoint_path
                    latest_iteration = checkpoint_iteration
                    
            except Exception:
                continue
        
        if latest_checkpoint_path is None or latest_iteration == 0:
            return 0
        
        current_time = time.time()
        checkpoint_age = current_time - latest_timestamp
        
        # If checkpoint older than 1 hour, consider it a new run
        if checkpoint_age > 3600:
            return 0
        
        return max(0, int(latest_iteration))
        
    except Exception:
        return 0


def get_evaluation_counter_from_file():
    """Get evaluation counter from file (cross-process)."""
    try:
        if os.path.exists(EVALUATION_COUNTER_FILE):
            file_age = time.time() - os.path.getmtime(EVALUATION_COUNTER_FILE)
            
            if file_age > 3600:
                try:
                    with open(EVALUATION_COUNTER_FILE, "w", encoding="utf-8") as f:
                        json.dump({"counter": 0}, f)
                except Exception:
                    pass
                return 0
            
            with open(EVALUATION_COUNTER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("counter", 0)
    except Exception:
        pass
    return 0


def increment_evaluation_counter_file():
    """Increment evaluation counter in file (cross-process)."""
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            counter = get_evaluation_counter_from_file()
            new_counter = counter + 1
            with open(EVALUATION_COUNTER_FILE, "w", encoding="utf-8") as f:
                json.dump({"counter": new_counter}, f)
            
            time.sleep(0.01)
            verify_counter = get_evaluation_counter_from_file()
            if verify_counter >= new_counter:
                return verify_counter
            
            time.sleep(retry_delay)
            
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                try:
                    counter = get_evaluation_counter_from_file()
                    new_counter = counter + 1
                    with open(EVALUATION_COUNTER_FILE, "w", encoding="utf-8") as f:
                        json.dump({"counter": new_counter}, f)
                    return new_counter
                except Exception:
                    return 1
    
    return 1


# ========================================================================================
# DATASET LOADING AND SPLITTING
# ========================================================================================

def load_prompt_config(prompt_path: str) -> Tuple[Dict, str]:
    """Load prompt from text file and dataset config from matching _dataset.yaml file."""
    global DATASET_CONFIG_PATH
    
    if DATASET_CONFIG_PATH is None:
        init_dataset_config_path()
    
    try:
        with open(prompt_path, "r", encoding="utf-8", errors="replace") as f:
            prompt = f.read().strip()
    except (UnicodeDecodeError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to load prompt file '{prompt_path}': {str(e)}") from e

    if not os.path.exists(DATASET_CONFIG_PATH):
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")

    try:
        with open(DATASET_CONFIG_PATH, "r", encoding="utf-8", errors="replace") as f:
            config = yaml.safe_load(f)
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load dataset config '{DATASET_CONFIG_PATH}': {str(e)}") from e

    return config, prompt


def split_dataset_train_val_test(dataset: List[Dict], train_ratio=0.7, validation_ratio=0.15, 
                                  test_ratio=0.15, seed=42) -> Tuple[List, List, List]:
    """
    Split dataset into train/validation/test sets with random shuffling.
    
    Args:
        dataset: List of examples
        train_ratio: Proportion for training (default: 0.7)
        validation_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple (train_dataset, validation_dataset, test_dataset)
    """
    total_size = len(dataset)
    
    # Shuffle dataset with fixed seed for reproducibility
    random.seed(seed)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(train_ratio * total_size)
    validation_size = int(validation_ratio * total_size)
    
    # Split indices
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]
    
    # Create splits
    train_dataset = [dataset[i] for i in train_indices]
    validation_dataset = [dataset[i] for i in validation_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    return train_dataset, validation_dataset, test_dataset


def load_livebench_if_dataset(config: Dict) -> Tuple[List, List, List]:
    """
    Load LiveBench Instruction Following dataset from HuggingFace.
    Automatically splits into train/validation/test.
    
    Args:
        config: Dataset configuration
    
    Returns:
        Tuple (train_dataset, validation_dataset, test_dataset)
    """
    cache_key = "livebench_if"
    
    # Check in-memory cache
    if cache_key in _DATASET_SPLITS:
        return (_DATASET_SPLITS[cache_key]["train"],
                _DATASET_SPLITS[cache_key]["validation"],
                _DATASET_SPLITS[cache_key]["test"])
    
    try:
        from datasets import load_dataset
        
        # Load from HuggingFace
        dataset_name = config.get("dataset_name", "livebench/instruction_following")
        split = config.get("split", "test")
        
        print(f"Loading LiveBench IF dataset from {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        
        # Convert to list
        dataset_list = list(dataset)
        
        # Filter by release date if specified
        livebench_release = config.get("livebench_release", None)
        if livebench_release:
            original_size = len(dataset_list)
            dataset_list = [q for q in dataset_list 
                          if q.get("livebench_release_date", "") <= livebench_release]
            print(f"Filtered by release {livebench_release}: {original_size} -> {len(dataset_list)}")
        
        # Split dataset
        train_ratio = config.get("train_ratio", 0.7)
        validation_ratio = config.get("validation_ratio", 0.15)
        test_ratio = config.get("test_ratio", 0.15)
        split_seed = config.get("split_seed", 42)
        
        train, val, test = split_dataset_train_val_test(
            dataset_list, train_ratio, validation_ratio, test_ratio, seed=split_seed
        )
        
        # Cache the splits
        _DATASET_SPLITS[cache_key] = {
            "train": train,
            "validation": val,
            "test": test
        }
        
        print(f"LiveBench IF split: Train={len(train)}, Validation={len(val)}, Test={len(test)}")
        
        return train, val, test
        
    except Exception as e:
        print(f"Error loading LiveBench dataset: {e}")
        traceback.print_exc()
        raise


# ========================================================================================
# EVALUATION FUNCTIONS
# ========================================================================================

def calculate_prompt_features(prompt: str) -> Tuple[int, float]:
    """
    Calculate custom features for MAP-Elites.
    
    Returns:
        tuple: (prompt_length, reasoning_sophistication_score)
    """
    # Feature 1: Prompt length (raw character count)
    prompt_length = len(prompt)

    # Feature 2: Reasoning sophistication score (continuous 0.0-1.0)
    prompt_lower = prompt.lower()
    sophistication_score = 0.0

    if len(prompt) >= 100:
        sophistication_score += 0.1

    # Check for few-shot examples
    has_example = (
        "example" in prompt_lower
        or prompt.count("####") >= 4
        or bool(re.search(r"problem:.*?solution:", prompt_lower, re.DOTALL))
    )

    # Check for Chain-of-Thought
    has_cot = (
        "step by step" in prompt_lower
        or "step-by-step" in prompt_lower
        or any(phrase in prompt_lower for phrase in ["think through", "reasoning", "explain your"])
        or bool(re.search(r"(first|then|next|finally)", prompt_lower))
    )

    has_directive = "solve" in prompt_lower or "calculate" in prompt_lower
    has_strict = "must" in prompt_lower or "exactly" in prompt_lower

    if has_example:
        sophistication_score += 0.6
        if has_cot:
            sophistication_score += 0.3
        elif len(prompt) > 1500:
            sophistication_score += 0.2
        else:
            sophistication_score += 0.1
    elif has_cot:
        sophistication_score += 0.4
        if has_strict:
            sophistication_score += 0.2
        elif len(prompt) > 500:
            sophistication_score += 0.15
        else:
            sophistication_score += 0.1
    else:
        if has_directive:
            sophistication_score += 0.2
        else:
            sophistication_score += 0.1

    sophistication_score = min(1.0, max(0.0, sophistication_score))

    return prompt_length, sophistication_score


def evaluate_single_example_livebench(
    prompt_template: str,
    example: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    max_tokens: int = 4096,
    temperature: float = 0.1
) -> Tuple[float, Dict]:
    """
    Evaluate a single LiveBench IF example using official evaluation scripts.
    
    Args:
        prompt_template: The prompt template with {instruction} placeholder
        example: Dataset example with turns, instruction_id_list, kwargs
        client: OpenAI client
        model_name: Model name to use
        max_tokens: Max tokens for response
        temperature: Temperature for generation
    
    Returns:
        Tuple (score, details_dict)
    """
    # Extract instruction from example
    instruction = example.get("turns", [""])[0]
    
    # Format the prompt
    try:
        formatted_prompt = prompt_template.format(instruction=instruction)
    except KeyError:
        # If {instruction} not in template, append instruction
        formatted_prompt = prompt_template + "\n\n" + instruction
    
    # Get model response
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to get response after {MAX_RETRIES} attempts: {e}")
                return 0.0, {"error": str(e)}
            time.sleep(1)
    
    if not response or not response.choices:
        return 0.0, {"error": "No response from model"}
    
    model_response = response.choices[0].message.content
    if model_response is None:
        return 0.0, {"error": "None content in response"}
    
    model_response = model_response.strip()
    
    # Determine task type and evaluate
    task = example.get("task", "")
    instruction_id_list = example.get("instruction_id_list", [])
    kwargs = example.get("kwargs", [])
    
    details = {
        "question_id": example.get("question_id", ""),
        "task": task,
        "instruction_ids": instruction_id_list,
        "response_preview": model_response[:200] + "..." if len(model_response) > 200 else model_response
    }
    
    if not LIVEBENCH_AVAILABLE:
        # Fallback: simple heuristic evaluation
        print("Warning: LiveBench not available, using fallback evaluation")
        score = 0.5  # Default score
        details["evaluation_method"] = "fallback"
        return score, details
    
    try:
        # Check if this is IFBench format (newer, harder constraints)
        if "ifbench" in task.lower() or any(":" in iid and not iid.startswith(("keywords:", "length_", "detectable_", "change_", "punctuation:", "startend:", "combination:")) for iid in instruction_id_list):
            # IFBench format
            inp = ifbench_lib.InputExample(
                key=example.get("question_id", example.get("key", 0)),
                instruction_id_list=instruction_id_list,
                prompt=instruction,
                kwargs=kwargs
            )
            
            result = ifbench_lib.test_instruction_following_strict(inp, model_response)
            score = score_results(result.follow_all_instructions, result.follow_instruction_list)
            details["evaluation_method"] = "ifbench"
            details["follow_all"] = result.follow_all_instructions
            details["follow_list"] = result.follow_instruction_list
            
        else:
            # IFEval format (original Google)
            # Clean kwargs
            clean_kwargs = [{k: v for k, v in d.items() if v is not None} for d in kwargs]
            
            inp = ifeval_main.InputExample(
                key=example.get("question_id", 0),
                instruction_id_list=instruction_id_list,
                prompt=instruction,
                kwargs=clean_kwargs
            )
            
            prompt_to_response = {instruction: model_response}
            result = ifeval_main.test_instruction_following_strict(inp, prompt_to_response)
            score = score_results(result.follow_all_instructions, result.follow_instruction_list)
            details["evaluation_method"] = "ifeval"
            details["follow_all"] = result.follow_all_instructions
            details["follow_list"] = result.follow_instruction_list
        
        return score, details
        
    except Exception as e:
        print(f"Evaluation error for question {example.get('question_id', 'unknown')}: {e}")
        details["error"] = str(e)
        return 0.0, details


def evaluate_prompt_on_samples(
    prompt: str,
    dataset: List[Dict],
    num_samples: int,
    client: OpenAI,
    model_name: str
) -> Tuple[float, int, int]:
    """
    Evaluate prompt on random samples from dataset.
    
    Args:
        prompt: Prompt template
        dataset: List of examples
        num_samples: Number of samples to evaluate
        client: OpenAI client
        model_name: Model name
    
    Returns:
        Tuple (accuracy, correct, total)
    """
    # Random sampling
    num_samples_actual = min(num_samples, len(dataset))
    random.seed()  # Use system seed for randomness
    indices = random.sample(range(len(dataset)), num_samples_actual)
    samples = [dataset[i] for i in indices]
    
    correct = 0
    total = 0
    
    for example in tqdm(samples, desc=f"Evaluating {num_samples_actual} samples"):
        score, details = evaluate_single_example_livebench(
            prompt, example, client, model_name, MAX_TOKENS
        )
        
        # Score >= 0.5 counts as correct (following convention)
        if score >= 0.5:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def get_adaptive_sample_size(stage: int, current_iteration: int, max_iterations: int = 100) -> int:
    """
    Adaptively increase sample size as evolution progresses.
    
    Args:
        stage: 1 for Stage 1, 2 for Stage 2
        current_iteration: Current iteration number
        max_iterations: Maximum iterations
    
    Returns:
        Sample size for this stage
    """
    progress = min(current_iteration / max_iterations, 1.0) if max_iterations > 0 else 0.0
    
    if stage == 1:
        # Stage 1: linear increase from 10 to 20
        min_samples = 10
        max_samples = 20
        return int(min_samples + (max_samples - min_samples) * progress)
    else:
        # Stage 2: linear increase from 20 to 60
        min_samples = 20
        max_samples = 60
        return int(min_samples + (max_samples - min_samples) * progress)


# ========================================================================================
# MAIN EVALUATION FUNCTIONS (Stage 1 and Stage 2)
# ========================================================================================

def evaluate_stage1(prompt_path: str) -> Dict[str, Any]:
    """
    Stage 1 evaluation: Quick evaluation with adaptive sample size.
    
    Args:
        prompt_path: Path to the prompt file
    
    Returns:
        Dictionary with combined_score and feature dimensions
    """
    global _MAX_ITERATIONS
    
    print("-" * 80)
    print("Starting Stage 1 evaluation (LiveBench IF)...")
    print("-" * 80)

    try:
        # Load max_iterations from config
        if _MAX_ITERATIONS == 100:
            try:
                with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
                    config_temp = yaml.safe_load(f)
                    _MAX_ITERATIONS = config_temp.get("max_iterations", 100)
            except Exception:
                pass
        
        # Increment evaluation counter
        evaluation_counter = increment_evaluation_counter_file()
        
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset
        train_dataset, _, _ = load_livebench_if_dataset(config)

        # Get current iteration from checkpoint
        current_iteration = get_current_iteration_from_checkpoint()
        
        # Adaptive sample size
        stage1_samples = get_adaptive_sample_size(1, current_iteration, _MAX_ITERATIONS)
        
        print(f"Stage 1: Evaluating {stage1_samples} random samples (adaptive, iteration {current_iteration}/{_MAX_ITERATIONS}, calls={evaluation_counter})...")

        # Run evaluation
        accuracy, correct, total = evaluate_prompt_on_samples(
            prompt, train_dataset, stage1_samples, test_model, TASK_MODEL_NAME
        )

        print(f"Stage 1 accuracy: {accuracy:.3f} ({correct}/{total})")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        print(f"Prompt features - Length: {prompt_length} chars, Reasoning sophistication: {reasoning_sophistication:.3f}")

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Stage 1 evaluation failed: {error_msg}")
        print("-" * 80)

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except Exception:
            prompt_length, reasoning_sophistication = 0, 0.0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate_stage2(prompt_path: str) -> Dict[str, Any]:
    """
    Stage 2 evaluation: Comprehensive evaluation with adaptive sample size.
    
    Args:
        prompt_path: Path to the prompt file
    
    Returns:
        Dictionary with combined_score and feature dimensions
    """
    global _MAX_ITERATIONS
    
    print("-" * 80)
    print("Starting Stage 2 evaluation (LiveBench IF)...")
    print("-" * 80)

    try:
        # Load max_iterations from config
        if _MAX_ITERATIONS == 100:
            try:
                with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
                    config_temp = yaml.safe_load(f)
                    _MAX_ITERATIONS = config_temp.get("max_iterations", 100)
            except Exception:
                pass
        
        # Get evaluation counter
        evaluation_counter = get_evaluation_counter_from_file()
        if evaluation_counter == 0:
            evaluation_counter = increment_evaluation_counter_file()
        
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset
        train_dataset, _, _ = load_livebench_if_dataset(config)

        # Get current iteration from checkpoint
        current_iteration = get_current_iteration_from_checkpoint()
        
        # Adaptive sample size
        stage2_samples = get_adaptive_sample_size(2, current_iteration, _MAX_ITERATIONS)
        
        print(f"Stage 2: Evaluating {stage2_samples} random samples (adaptive, iteration {current_iteration}/{_MAX_ITERATIONS}, calls={evaluation_counter})...")

        # Run evaluation
        accuracy, correct, total = evaluate_prompt_on_samples(
            prompt, train_dataset, stage2_samples, test_model, TASK_MODEL_NAME
        )

        print(f"Stage 2 accuracy: {accuracy:.3f} ({correct}/{total})")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        print(f"Prompt features - Length: {prompt_length} chars, Reasoning sophistication: {reasoning_sophistication:.3f}")

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Stage 2 evaluation failed: {error_msg}")
        print("-" * 80)

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except Exception:
            prompt_length, reasoning_sophistication = 0, 0.0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate(prompt_path: str) -> Dict[str, Any]:
    """
    Main evaluation function - for backwards compatibility.
    Calls evaluate_stage2 for full evaluation.
    
    Args:
        prompt_path: Path to the prompt file
    
    Returns:
        Dictionary with combined_score metric
    """
    return evaluate_stage2(prompt_path)

