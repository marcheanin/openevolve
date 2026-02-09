"""
Evaluator for LiveBench IF prompt optimization with SEPARATE models for evolution and evaluation.

This evaluator supports using different models:
- Evolution model: Used by OpenEvolve to generate improved prompts (from config.yaml llm section)
- Evaluation model: Used to execute prompts on the dataset (from config.yaml evaluation section)

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
_DATASET_SPLITS = {}
_MAX_ITERATIONS = 100

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".evaluation_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

EVALUATION_COUNTER_FILE = os.path.join(CACHE_DIR, "evaluation_counter.json")
DATASET_SPLITS_CACHE_FILE = os.path.join(CACHE_DIR, "dataset_splits_cache.pkl")

# ========================================================================================
# CONFIGURATION LOADING WITH SEPARATE MODELS SUPPORT
# ========================================================================================
config_file = "config_separate_models.yaml"
if not os.path.exists(os.path.join(os.path.dirname(__file__), config_file)):
    config_file = "config.yaml"

config_path = os.path.join(os.path.dirname(__file__), config_file)
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ========================================================================================
# SETUP EVALUATION MODEL (separate from evolution model)
# ========================================================================================
evaluation_config = config.get("evaluation", {})

if evaluation_config:
    if not os.environ.get("_EVAL_MODEL_INIT_LOGGED"):
        print("="*80)
        print("Using SEPARATE evaluation model configuration")
        print("="*80)
        os.environ["_EVAL_MODEL_INIT_LOGGED"] = "1"
    
    eval_api_base = evaluation_config.get("api_base")
    if eval_api_base is None:
        llm_config = config.get("llm", {})
        eval_api_base = llm_config.get("api_base", "http://localhost:1234/v1")
    
    TASK_MODEL_NAME = evaluation_config.get("model", "default-model")
    eval_temperature = evaluation_config.get("temperature", 0.1)
    MAX_TOKENS = evaluation_config.get("max_tokens", 4096)
    eval_timeout = evaluation_config.get("timeout", 120)
    
    if not os.environ.get("_EVAL_MODEL_INIT_LOGGED"):
        print(f"Evaluation Model: {TASK_MODEL_NAME}")
        print(f"Evaluation API Base: {eval_api_base}")
        print("="*80)
else:
    llm_config = config.get("llm", {})
    eval_api_base = llm_config.get("api_base", "http://localhost:1234/v1")
    
    models = llm_config.get("models", [])
    if models:
        TASK_MODEL_NAME = models[0].get("name", "default-model")
    else:
        TASK_MODEL_NAME = llm_config.get("primary_model", "default-model")
    
    eval_temperature = 0.1
    MAX_TOKENS = llm_config.get("max_tokens", 4096)

# Stage-specific models (optional)
eval_stage1_model = evaluation_config.get("stage1_model") if evaluation_config else None
eval_stage2_model = evaluation_config.get("stage2_model") if evaluation_config else None

evaluator_config = config.get("evaluator", {})
MAX_RETRIES = evaluator_config.get("max_retries", 3)

test_model = OpenAI(base_url=eval_api_base)

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
# CROSS-PROCESS CACHING (same as evaluator.py)
# ========================================================================================

def get_current_iteration_from_checkpoint():
    """Get current iteration from most recent checkpoint."""
    try:
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(evaluator_dir, "openevolve_output", "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            return 0
        
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
                    latest_iteration = checkpoint_iteration
                    
            except Exception:
                continue
        
        if latest_iteration == 0:
            return 0
        
        current_time = time.time()
        checkpoint_age = current_time - latest_timestamp
        
        if checkpoint_age > 3600:
            return 0
        
        return max(0, int(latest_iteration))
        
    except Exception:
        return 0


def get_evaluation_counter_from_file():
    """Get evaluation counter from file."""
    try:
        if os.path.exists(EVALUATION_COUNTER_FILE):
            file_age = time.time() - os.path.getmtime(EVALUATION_COUNTER_FILE)
            if file_age > 3600:
                with open(EVALUATION_COUNTER_FILE, "w", encoding="utf-8") as f:
                    json.dump({"counter": 0}, f)
                return 0
            
            with open(EVALUATION_COUNTER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("counter", 0)
    except Exception:
        pass
    return 0


def increment_evaluation_counter_file():
    """Increment evaluation counter in file."""
    for attempt in range(10):
        try:
            counter = get_evaluation_counter_from_file()
            new_counter = counter + 1
            with open(EVALUATION_COUNTER_FILE, "w", encoding="utf-8") as f:
                json.dump({"counter": new_counter}, f)
            time.sleep(0.01)
            return new_counter
        except Exception:
            time.sleep(0.1)
    return 1


# ========================================================================================
# DATASET LOADING AND SPLITTING
# ========================================================================================

def load_prompt_config(prompt_path: str) -> Tuple[Dict, str]:
    """Load prompt and dataset config."""
    global DATASET_CONFIG_PATH
    
    if DATASET_CONFIG_PATH is None:
        init_dataset_config_path()
    
    with open(prompt_path, "r", encoding="utf-8", errors="replace") as f:
        prompt = f.read().strip()

    if not os.path.exists(DATASET_CONFIG_PATH):
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")

    with open(DATASET_CONFIG_PATH, "r", encoding="utf-8", errors="replace") as f:
        config = yaml.safe_load(f)

    return config, prompt


def split_dataset_train_val_test(dataset: List[Dict], train_ratio=0.7, validation_ratio=0.15, 
                                  test_ratio=0.15, seed=42) -> Tuple[List, List, List]:
    """Split dataset into train/validation/test with random shuffling."""
    total_size = len(dataset)
    
    random.seed(seed)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_size = int(train_ratio * total_size)
    validation_size = int(validation_ratio * total_size)
    
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]
    
    train_dataset = [dataset[i] for i in train_indices]
    validation_dataset = [dataset[i] for i in validation_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    return train_dataset, validation_dataset, test_dataset


def load_livebench_if_dataset(config: Dict) -> Tuple[List, List, List]:
    """Load and split LiveBench IF dataset."""
    cache_key = "livebench_if"
    
    if cache_key in _DATASET_SPLITS:
        return (_DATASET_SPLITS[cache_key]["train"],
                _DATASET_SPLITS[cache_key]["validation"],
                _DATASET_SPLITS[cache_key]["test"])
    
    try:
        from datasets import load_dataset
        
        dataset_name = config.get("dataset_name", "livebench/instruction_following")
        split = config.get("split", "test")
        
        print(f"Loading LiveBench IF dataset from {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        dataset_list = list(dataset)
        
        livebench_release = config.get("livebench_release", None)
        if livebench_release:
            original_size = len(dataset_list)
            dataset_list = [q for q in dataset_list 
                          if q.get("livebench_release_date", "") <= livebench_release]
            print(f"Filtered by release {livebench_release}: {original_size} -> {len(dataset_list)}")
        
        train_ratio = config.get("train_ratio", 0.7)
        validation_ratio = config.get("validation_ratio", 0.15)
        test_ratio = config.get("test_ratio", 0.15)
        split_seed = config.get("split_seed", 42)
        
        train, val, test = split_dataset_train_val_test(
            dataset_list, train_ratio, validation_ratio, test_ratio, seed=split_seed
        )
        
        _DATASET_SPLITS[cache_key] = {"train": train, "validation": val, "test": test}
        print(f"LiveBench IF split: Train={len(train)}, Validation={len(val)}, Test={len(test)}")
        
        return train, val, test
        
    except Exception as e:
        print(f"Error loading LiveBench dataset: {e}")
        traceback.print_exc()
        raise


# ========================================================================================
# FEATURE CALCULATION
# ========================================================================================

def calculate_prompt_features(prompt: str) -> Tuple[int, float]:
    """Calculate custom features for MAP-Elites."""
    prompt_length = len(prompt)
    prompt_lower = prompt.lower()
    sophistication_score = 0.0

    if len(prompt) >= 100:
        sophistication_score += 0.1

    has_example = (
        "example" in prompt_lower
        or prompt.count("####") >= 4
        or bool(re.search(r"problem:.*?solution:", prompt_lower, re.DOTALL))
    )

    has_cot = (
        "step by step" in prompt_lower
        or "step-by-step" in prompt_lower
        or any(phrase in prompt_lower for phrase in ["think through", "reasoning", "explain your"])
    )

    has_directive = "solve" in prompt_lower or "calculate" in prompt_lower
    has_strict = "must" in prompt_lower or "exactly" in prompt_lower

    if has_example:
        sophistication_score += 0.6
        sophistication_score += 0.3 if has_cot else (0.2 if len(prompt) > 1500 else 0.1)
    elif has_cot:
        sophistication_score += 0.4
        sophistication_score += 0.2 if has_strict else (0.15 if len(prompt) > 500 else 0.1)
    else:
        sophistication_score += 0.2 if has_directive else 0.1

    return prompt_length, min(1.0, max(0.0, sophistication_score))


# ========================================================================================
# EVALUATION FUNCTIONS
# ========================================================================================

def evaluate_single_example_livebench(
    prompt_template: str,
    example: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    max_tokens: int = 4096,
    temperature: float = 0.1
) -> Tuple[float, Dict]:
    """Evaluate a single LiveBench IF example."""
    instruction = example.get("turns", [""])[0]
    
    try:
        formatted_prompt = prompt_template.format(instruction=instruction)
    except KeyError:
        formatted_prompt = prompt_template + "\n\n" + instruction
    
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
                return 0.0, {"error": str(e)}
            time.sleep(1)
    
    if not response or not response.choices:
        return 0.0, {"error": "No response"}
    
    model_response = response.choices[0].message.content
    if model_response is None:
        return 0.0, {"error": "None content"}
    
    model_response = model_response.strip()
    
    task = example.get("task", "")
    instruction_id_list = example.get("instruction_id_list", [])
    kwargs = example.get("kwargs", [])
    
    details = {
        "question_id": example.get("question_id", ""),
        "task": task,
        "instruction_ids": instruction_id_list,
    }
    
    if not LIVEBENCH_AVAILABLE:
        return 0.5, {"evaluation_method": "fallback", **details}
    
    try:
        if "ifbench" in task.lower() or any(":" in iid and not iid.startswith(("keywords:", "length_", "detectable_", "change_", "punctuation:", "startend:", "combination:")) for iid in instruction_id_list):
            inp = ifbench_lib.InputExample(
                key=example.get("question_id", example.get("key", 0)),
                instruction_id_list=instruction_id_list,
                prompt=instruction,
                kwargs=kwargs
            )
            result = ifbench_lib.test_instruction_following_strict(inp, model_response)
            score = score_results(result.follow_all_instructions, result.follow_instruction_list)
            details["evaluation_method"] = "ifbench"
        else:
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
        return score, details
        
    except Exception as e:
        details["error"] = str(e)
        return 0.0, details


def evaluate_prompt_on_samples(prompt: str, dataset: List[Dict], num_samples: int,
                                client: OpenAI, model_name: str) -> Tuple[float, int, int]:
    """Evaluate prompt on random samples."""
    num_samples_actual = min(num_samples, len(dataset))
    random.seed()
    indices = random.sample(range(len(dataset)), num_samples_actual)
    samples = [dataset[i] for i in indices]
    
    correct = 0
    total = 0
    
    for example in tqdm(samples, desc=f"Evaluating {num_samples_actual} samples"):
        score, _ = evaluate_single_example_livebench(prompt, example, client, model_name, MAX_TOKENS)
        if score >= 0.5:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def get_adaptive_sample_size(stage: int, current_iteration: int, max_iterations: int = 100) -> int:
    """Adaptively increase sample size."""
    progress = min(current_iteration / max_iterations, 1.0) if max_iterations > 0 else 0.0
    
    if stage == 1:
        return int(10 + 10 * progress)
    else:
        return int(20 + 40 * progress)


# ========================================================================================
# MAIN EVALUATION FUNCTIONS
# ========================================================================================

def evaluate_stage1(prompt_path: str) -> Dict[str, Any]:
    """Stage 1 quick evaluation."""
    global _MAX_ITERATIONS
    
    print("-" * 80)
    print("Starting Stage 1 evaluation (LiveBench IF - Separate Models)...")
    print("-" * 80)

    try:
        if _MAX_ITERATIONS == 100:
            try:
                with open(os.path.join(os.path.dirname(__file__), "config_separate_models.yaml"), "r") as f:
                    config_temp = yaml.safe_load(f)
                    _MAX_ITERATIONS = config_temp.get("max_iterations", 100)
            except Exception:
                pass
        
        evaluation_counter = increment_evaluation_counter_file()
        config, prompt = load_prompt_config(prompt_path)
        
        train_dataset, _, _ = load_livebench_if_dataset(config)
        current_iteration = get_current_iteration_from_checkpoint()
        
        # Use stage-specific model if configured
        model_to_use = eval_stage1_model if eval_stage1_model else TASK_MODEL_NAME
        
        stage1_samples = get_adaptive_sample_size(1, current_iteration, _MAX_ITERATIONS)
        print(f"Stage 1: {stage1_samples} samples, model={model_to_use}, iteration={current_iteration}/{_MAX_ITERATIONS}")

        accuracy, correct, total = evaluate_prompt_on_samples(
            prompt, train_dataset, stage1_samples, test_model, model_to_use
        )

        print(f"Stage 1 accuracy: {accuracy:.3f} ({correct}/{total})")
        
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
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
    """Stage 2 comprehensive evaluation."""
    global _MAX_ITERATIONS
    
    print("-" * 80)
    print("Starting Stage 2 evaluation (LiveBench IF - Separate Models)...")
    print("-" * 80)

    try:
        if _MAX_ITERATIONS == 100:
            try:
                with open(os.path.join(os.path.dirname(__file__), "config_separate_models.yaml"), "r") as f:
                    config_temp = yaml.safe_load(f)
                    _MAX_ITERATIONS = config_temp.get("max_iterations", 100)
            except Exception:
                pass
        
        evaluation_counter = get_evaluation_counter_from_file()
        if evaluation_counter == 0:
            evaluation_counter = increment_evaluation_counter_file()
        
        config, prompt = load_prompt_config(prompt_path)
        train_dataset, _, _ = load_livebench_if_dataset(config)
        current_iteration = get_current_iteration_from_checkpoint()
        
        # Use stage-specific model if configured
        model_to_use = eval_stage2_model if eval_stage2_model else TASK_MODEL_NAME
        
        stage2_samples = get_adaptive_sample_size(2, current_iteration, _MAX_ITERATIONS)
        print(f"Stage 2: {stage2_samples} samples, model={model_to_use}, iteration={current_iteration}/{_MAX_ITERATIONS}")

        accuracy, correct, total = evaluate_prompt_on_samples(
            prompt, train_dataset, stage2_samples, test_model, model_to_use
        )

        print(f"Stage 2 accuracy: {accuracy:.3f} ({correct}/{total})")
        
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        print(f"Stage 2 evaluation failed: {e}")
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
    """Main evaluation function."""
    return evaluate_stage2(prompt_path)

