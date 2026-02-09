"""
Evaluator for HuggingFace dataset-based prompt optimization.
"""

import re
import traceback
import yaml
import os
import time
import random
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

# Глобальные переменные для разделения датасета
_DATASET_SPLITS = {}  # Кэш для train/validation/test разделений
_EVALUATION_COUNTER = 0  # Счетчик вызовов оценки для адаптивного размера выборки
_MAX_ITERATIONS = 100  # Максимальное количество итераций (берется из конфига при первом вызове)

# Read config.yaml to get model settings
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Get model settings from config
llm_config = config.get("llm", {})
api_base = llm_config.get("api_base", "http://localhost:1234/v1")

# Handle both single model and model list configurations
models = llm_config.get("models", [])
if models:
    # Use first model from list
    TASK_MODEL_NAME = models[0].get("name", "default-model")
else:
    # Fallback to direct model specification
    TASK_MODEL_NAME = llm_config.get("primary_model", "default-model")

# Get evaluator settings
evaluator_config = config.get("evaluator", {})
MAX_RETRIES = evaluator_config.get("max_retries", 3)

# Get max_tokens from LLM config
MAX_TOKENS = llm_config.get("max_tokens", 16000)
print(f"Using max_tokens: {MAX_TOKENS}")

# Initialize OpenAI client once for all evaluations
test_model = OpenAI(base_url=api_base)
print(f"Initialized OpenAI client with model: {TASK_MODEL_NAME}")

# Determine which dataset to use based on the OPENEVOLVE_PROMPT environment variable
import sys

prompt_file = os.environ.get("OPENEVOLVE_PROMPT")
if not prompt_file:
    # Default to a generic dataset config if not using the wrapper script
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG_PATH = os.path.join(evaluator_dir, "dataset_settings.yaml")
    print("Warning: OPENEVOLVE_PROMPT not set. Using default dataset_settings.yaml")
else:
    basename = os.path.basename(prompt_file)
    dataset_filename = basename.replace("_prompt.txt", "_prompt_dataset.yaml").replace(
        ".txt", "_dataset.yaml"
    )
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG_PATH = os.path.join(evaluator_dir, dataset_filename)
    print(f"Dataset configuration: {dataset_filename}")


def calculate_prompt_features(prompt):
    """
    Calculate custom features for MAP-Elites

    IMPORTANT: Returns raw continuous values, not bin indices.
    The database handles all scaling and binning automatically.

    Returns:
        tuple: (prompt_length, reasoning_sophistication_score)
        - prompt_length: Actual character count
        - reasoning_sophistication_score: Continuous score 0.0-1.0
    """
    # Feature 1: Prompt length (raw character count)
    prompt_length = len(prompt)

    # Feature 2: Reasoning sophistication score (continuous 0.0-1.0)
    prompt_lower = prompt.lower()
    sophistication_score = 0.0

    # Base scoring
    if len(prompt) >= 100:
        sophistication_score += 0.1  # Has substantial content

    # Check for few-shot examples (high sophistication)
    has_example = (
        "example" in prompt_lower
        or prompt.count("####") >= 4
        or bool(re.search(r"problem:.*?solution:", prompt_lower, re.DOTALL))
    )

    # Check for Chain-of-Thought (CoT) indicators
    has_cot = (
        "step by step" in prompt_lower
        or "step-by-step" in prompt_lower
        or any(phrase in prompt_lower for phrase in ["think through", "reasoning", "explain your"])
        or bool(re.search(r"(first|then|next|finally)", prompt_lower))
    )

    # Check for directive language
    has_directive = "solve" in prompt_lower or "calculate" in prompt_lower

    # Check for strict language
    has_strict = "must" in prompt_lower or "exactly" in prompt_lower

    # Calculate sophistication score
    if has_example:
        sophistication_score += 0.6  # Few-shot examples are sophisticated
        if has_cot:
            sophistication_score += 0.3  # Few-shot + CoT is most sophisticated
        elif len(prompt) > 1500:
            sophistication_score += 0.2  # Extensive few-shot
        else:
            sophistication_score += 0.1  # Basic few-shot
    elif has_cot:
        sophistication_score += 0.4  # Chain-of-thought
        if has_strict:
            sophistication_score += 0.2  # Strict CoT
        elif len(prompt) > 500:
            sophistication_score += 0.15  # Detailed CoT
        else:
            sophistication_score += 0.1  # Basic CoT
    else:
        # Basic prompts
        if has_directive:
            sophistication_score += 0.2  # Direct instruction
        else:
            sophistication_score += 0.1  # Simple prompt

    # Ensure score is within 0.0-1.0 range
    sophistication_score = min(1.0, max(0.0, sophistication_score))

    return prompt_length, sophistication_score


def load_prompt_config(prompt_path):
    """Load the prompt from text file and dataset config from matching _dataset.yaml file."""
    try:
        # Load prompt from text file
        with open(prompt_path, "r", encoding="utf-8", errors="replace") as f:
            prompt = f.read().strip()
    except (UnicodeDecodeError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to load prompt file '{prompt_path}': {str(e)}") from e

    # Load the configuration (already determined from environment variable)
    if not os.path.exists(DATASET_CONFIG_PATH):
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")

    try:
        with open(DATASET_CONFIG_PATH, "r", encoding="utf-8", errors="replace") as f:
            config = yaml.safe_load(f)
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load dataset config '{DATASET_CONFIG_PATH}': {str(e)}") from e

    return config, prompt


def split_dataset_train_val_test(dataset, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Разделяет датасет на train/validation/test выборки.
    
    Args:
        dataset: HuggingFace dataset
        train_ratio: Доля тренировочной выборки (default: 0.7)
        validation_ratio: Доля валидационной выборки (default: 0.15)
        test_ratio: Доля тестовой выборки (default: 0.15)
        seed: Seed для воспроизводимости (default: 42)
    
    Returns:
        Tuple (train_dataset, validation_dataset, test_dataset)
    """
    if not hasattr(dataset, "__len__"):
        raise ValueError("Cannot split streaming dataset. Set streaming=False in config.")
    
    total_size = len(dataset)
    
    # Перемешиваем датасет для случайности
    dataset_shuffled = dataset.shuffle(seed=seed)
    
    # Вычисляем размеры выборок
    train_size = int(train_ratio * total_size)
    validation_size = int(validation_ratio * total_size)
    test_size = total_size - train_size - validation_size
    
    # Разделяем
    train_dataset = dataset_shuffled.select(range(train_size))
    validation_dataset = dataset_shuffled.select(range(train_size, train_size + validation_size))
    test_dataset = dataset_shuffled.select(range(train_size + validation_size, total_size))
    
    print(f"Dataset split: Train={len(train_dataset)}, Validation={len(validation_dataset)}, Test={len(test_dataset)}")
    
    return train_dataset, validation_dataset, test_dataset


def get_dataset_splits(dataset_name, split="validation"):
    """
    Получить разделенные датасеты (train/validation/test) из кэша.
    
    Args:
        dataset_name: Имя датасета
        split: Исходный split (обычно "validation" для HotpotQA)
    
    Returns:
        Dict с ключами "train", "validation", "test" или None если не найдено
    """
    cache_key = f"{dataset_name}_{split}"
    return _DATASET_SPLITS.get(cache_key)


def load_hf_dataset(config):
    """
    Load HuggingFace dataset based on configuration.
    For HotpotQA: automatically splits into train/validation/test (70/15/15).
    """
    dataset_name = config["dataset_name"]
    dataset_config = config.get("dataset_config", None)
    split = config.get("split", "validation")
    trust_remote_code = config.get("trust_remote_code", True)
    is_hotpotqa = config.get("is_hotpotqa", False)
    
    # Для HotpotQA используем разделение на train/validation/test
    if is_hotpotqa:
        # Проверяем кэш
        cache_key = f"{dataset_name}_{split}"
        if dataset_config:
            cache_key = f"{dataset_name}_{dataset_config}_{split}"
        if cache_key in _DATASET_SPLITS:
            print(f"Using cached dataset splits for {dataset_name}")
            return _DATASET_SPLITS[cache_key]["train"]  # Возвращаем train для эволюции
    
    print(f"Loading dataset: {dataset_name}")

    # Для HotpotQA всегда используем non-streaming mode (необходимо для разделения)
    if dataset_name == "hotpot_qa" or is_hotpotqa:
        print("Using non-streaming mode for HotpotQA to avoid PyArrow issues and enable splitting")
        streaming = False
    else:
        # For other datasets, use streaming if not specified
        streaming = config.get("streaming", True)

    try:
        # Try to load the specified split
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name, split=split, trust_remote_code=trust_remote_code, streaming=streaming
            )
    except:
        # Fallback to train split if test is not available
        print(f"Split '{split}' not found, falling back to 'train'")
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split="train",
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split="train",
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )

    # Для HotpotQA: разделяем на train/validation/test
    if is_hotpotqa and not streaming and hasattr(dataset, "__len__"):
        print("\n" + "="*80)
        print("HotpotQA dataset detected: Splitting into Train/Validation/Test (70/15/15)")
        print("="*80)
        
        train_ratio = config.get("train_ratio", 0.7)
        validation_ratio = config.get("validation_ratio", 0.15)
        test_ratio = config.get("test_ratio", 0.15)
        split_seed = config.get("split_seed", 42)
        
        train_dataset, validation_dataset, test_dataset = split_dataset_train_val_test(
            dataset, train_ratio, validation_ratio, test_ratio, seed=split_seed
        )
        
        # Сохраняем в кэш
        cache_key = f"{dataset_name}_{split}"
        if dataset_config:
            cache_key = f"{dataset_name}_{dataset_config}_{split}"
        _DATASET_SPLITS[cache_key] = {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        }
        
        print(f"✓ Train set: {len(train_dataset)} examples (for evolution)")
        print(f"✓ Validation set: {len(validation_dataset)} examples (for overfitting control)")
        print(f"✓ Test set: {len(test_dataset)} examples (for final evaluation only)")
        print("="*80 + "\n")
        
        # Возвращаем train для эволюции
        return train_dataset

    # Print dataset info
    if hasattr(dataset, "__len__"):
        print(f"Dataset loaded with {len(dataset)} examples")
    else:
        print(f"Dataset loaded (streaming mode)")

    return dataset


def evaluate_prompt(prompt, dataset, config, num_samples):
    """Evaluate a prompt on a subset of the dataset."""
    input_field = config["input_field"]
    target_field = config["target_field"]

    # Check dataset type
    dataset_name = config.get("dataset_name", "").lower()
    is_emotion = "emotion" in dataset_name
    is_gsm8k = "gsm8k" in dataset_name
    is_hotpotqa = config.get("is_hotpotqa", False)
    is_ifeval = config.get("is_ifeval", False)
    is_hover = config.get("is_hover", False)

    # Sample from dataset - handle both streaming and non-streaming
    # ИСПОЛЬЗУЕМ СЛУЧАЙНУЮ ВЫБОРКУ вместо фиксированной
    if hasattr(dataset, "take"):
        # Streaming dataset - нельзя сделать случайную выборку, используем первые N
        samples = dataset.take(num_samples)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples", total=num_samples)
    else:
        # Non-streaming dataset - СЛУЧАЙНАЯ выборка
        dataset_size = len(dataset)
        num_samples_actual = min(num_samples, dataset_size)
        
        # Случайная выборка с фиксированным seed для воспроизводимости внутри одного вызова
        # Но seed меняется при каждом вызове для разнообразия
        random.seed()  # Используем системный seed для случайности
        indices = random.sample(range(dataset_size), num_samples_actual)
        samples = dataset.select(indices)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples_actual} random samples")

    correct = 0
    total = 0

    for example in sample_iter:
        input_text = example[input_field]
        expected = example[target_field]

        # Prepare the prompt with appropriate formatting
        if is_hotpotqa:
            # Format context from paragraphs
            context_items = example.get("context", {})
            context_text = ""
            if "title" in context_items and "sentences" in context_items:
                # Handle the specific structure of HotpotQA
                for i, (title, sentences) in enumerate(
                    zip(context_items["title"], context_items["sentences"])
                ):
                    context_text += f"Paragraph {i+1} ({title}):\n"
                    context_text += " ".join(sentences) + "\n\n"
            formatted_prompt = prompt.format(context=context_text.strip(), question=input_text)
        elif is_ifeval:
            # IFEval uses 'prompt' field directly
            formatted_prompt = prompt.format(instruction=input_text)
        elif is_hover:
            # HoVer uses claim field
            formatted_prompt = prompt.format(claim=input_text)
        else:
            # Default formatting for other datasets
            formatted_prompt = prompt.format(input_text=input_text)

        # Prepare the message for the LLM
        messages = [{"role": "user", "content": formatted_prompt}]

        # Call the LLM with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                # Use max_tokens from config
                response = test_model.chat.completions.create(
                    model=TASK_MODEL_NAME,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=MAX_TOKENS,
                )
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get response after {MAX_RETRIES} attempts: {e}")
                    raise e
                time.sleep(1)

        # Handle potential None response
        if not response:
            print(f"Warning: No response object from LLM")
            total += 1  # Count as incorrect
            continue

        if not response.choices:
            print(f"Warning: No choices in response from LLM")
            total += 1  # Count as incorrect
            continue

        if not response.choices[0].message:
            print(f"Warning: No message in response choice")
            total += 1  # Count as incorrect
            continue

        output_text = response.choices[0].message.content
        if output_text is None:
            print(f"Warning: None content in LLM response")
            print(f"Full response: {response}")
            total += 1  # Count as incorrect
            continue

        output_text = output_text.strip()

        # Extract prediction from output
        try:
            if is_gsm8k:
                # For GSM8K, extract the numeric answer after ####
                # First, extract the expected answer from the ground truth
                expected_answer = expected.split("####")[-1].strip()
                try:
                    expected_number = float(expected_answer.replace(",", ""))
                except:
                    print(f"Warning: Could not parse expected answer: {expected_answer}")
                    total += 1
                    continue

                # Extract prediction from model output
                prediction = None
                if "####" in output_text:
                    predicted_answer = output_text.split("####")[-1].strip()
                    # Extract just the number, removing any extra text like $ signs
                    import re

                    numbers = re.findall(r"-?\$?[\d,]+\.?\d*", predicted_answer)
                    if numbers:
                        try:
                            # Remove $ and , from the number
                            number_str = numbers[0].replace("$", "").replace(",", "")
                            prediction = float(number_str)
                        except:
                            pass

                # If we found a prediction, check if it matches
                if prediction is not None:
                    # Check if answers match (with small tolerance for floats)
                    if abs(prediction - expected_number) < 0.001:
                        correct += 1

                total += 1
                continue  # Skip the general case to avoid double counting

            elif is_hotpotqa:
                # For HotpotQA, do exact match comparison (case-insensitive)
                output_lower = output_text.lower().strip()
                expected_lower = str(expected).lower().strip()

                # Remove common punctuation for better matching
                output_lower = output_lower.rstrip(".,!?;:")
                expected_lower = expected_lower.rstrip(".,!?;:")

                if output_lower == expected_lower:
                    correct += 1
                elif expected_lower in output_lower:
                    # Partial credit if answer is contained in response
                    correct += 1

                total += 1
                continue

            elif is_ifeval:
                # For IFEval, we need more complex evaluation
                # For now, do basic keyword matching
                # Note: Full IFEval requires checking multiple constraints
                output_lower = output_text.lower()

                # Simple heuristic: check if response seems to follow instruction format
                if len(output_text.strip()) > 10:  # Non-trivial response
                    correct += 1  # Simplified - real IFEval needs constraint checking

                total += 1
                continue

            elif is_hover:
                # For HoVer, check if prediction matches SUPPORTED/NOT_SUPPORTED
                output_upper = output_text.upper()
                expected_upper = str(expected).upper()

                # Look for the verdict in the output
                if "SUPPORTED" in output_upper and "NOT" not in output_upper.replace(
                    "NOT SUPPORTED", ""
                ):
                    prediction = "SUPPORTED"
                elif "NOT SUPPORTED" in output_upper or "NOT_SUPPORTED" in output_upper:
                    prediction = "NOT_SUPPORTED"
                else:
                    prediction = None

                if prediction == expected_upper:
                    correct += 1

                total += 1
                continue

            elif is_emotion:
                # For emotion classification (0-5)
                numbers = re.findall(r"\b[0-5]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])  # Use the last number found
                else:
                    # Try to infer from emotion keywords
                    output_lower = output_text.lower()
                    emotion_map = {
                        "sadness": 0,
                        "sad": 0,
                        "joy": 1,
                        "happy": 1,
                        "happiness": 1,
                        "love": 2,
                        "anger": 3,
                        "angry": 3,
                        "fear": 4,
                        "afraid": 4,
                        "scared": 4,
                        "surprise": 5,
                        "surprised": 5,
                    }
                    prediction = -1
                    for emotion, label in emotion_map.items():
                        if emotion in output_lower:
                            prediction = label
                            break
            else:
                # For sentiment classification (0-1)
                numbers = re.findall(r"\b[01]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])  # Use the last number found
                else:
                    # Try to infer from keywords
                    output_lower = output_text.lower()
                    if "positive" in output_lower:
                        prediction = 1
                    elif "negative" in output_lower:
                        prediction = 0
                    else:
                        prediction = -1  # Invalid prediction

            if prediction == expected:
                correct += 1

            total += 1

        except Exception as e:
            print(f"Error parsing response '{output_text}': {e}")
            total += 1  # Count as incorrect

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def get_adaptive_sample_size(stage, counter, max_iterations=100):
    """
    Адаптивно увеличиваем размер выборки по мере прогресса эволюции.
    
    Args:
        stage: 1 для Stage 1, 2 для Stage 2
        counter: Счетчик вызовов оценки (приблизительный номер итерации)
        max_iterations: Максимальное количество итераций
    
    Returns:
        Размер выборки для данного этапа
    """
    # Нормализуем прогресс (0.0 - 1.0)
    progress = min(counter / max_iterations, 1.0) if max_iterations > 0 else 0.0
    
    if stage == 1:
        # Stage 1: от 10 до 20 примеров
        if progress < 0.3:
            return 10  # Ранние итерации: быстрая оценка
        elif progress < 0.7:
            return 15  # Средние итерации
        else:
            return 20  # Поздние итерации: более надежная оценка
    else:  # stage == 2
        # Stage 2: от 20 до 80 примеров
        if progress < 0.3:
            return 20  # Ранние итерации
        elif progress < 0.7:
            return 50  # Средние итерации
        else:
            return 80  # Поздние итерации: тщательная оценка


def evaluate_stage1(prompt_path):
    """
    Stage 1 evaluation: Quick evaluation with adaptive sample size

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    global _EVALUATION_COUNTER, _MAX_ITERATIONS
    
    print("-" * 80)
    print("Starting Stage 1 evaluation...")
    print("-" * 80)

    try:
        # Загружаем max_iterations из конфига при первом вызове
        if _MAX_ITERATIONS == 100:
            try:
                with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
                    config_temp = yaml.safe_load(f)
                    _MAX_ITERATIONS = config_temp.get("max_iterations", 100)
            except Exception:
                pass  # Используем значение по умолчанию
        
        # Увеличиваем счетчик вызовов
        _EVALUATION_COUNTER += 1
        
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset (будет автоматически разделен для HotpotQA)
        dataset = load_hf_dataset(config)

        # Адаптивный размер выборки
        stage1_samples = get_adaptive_sample_size(1, _EVALUATION_COUNTER, _MAX_ITERATIONS)
        
        print(f"Stage 1: Evaluating {stage1_samples} random samples (adaptive, iteration ~{_EVALUATION_COUNTER})...")

        # Run evaluation
        accuracy, correct, total = evaluate_prompt(prompt, dataset, config, stage1_samples)

        print(f"Stage 1 accuracy: {accuracy:.3f} ({correct}/{total})")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        print(
            f"Prompt features - Length: {prompt_length} chars, Reasoning sophistication: {reasoning_sophistication:.3f}"
        )

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        error_msg = str(e)
        # Краткое сообщение об ошибке без полного traceback
        print(f"Stage 1 evaluation failed: {error_msg}")
        print("-" * 80)

        # Always return feature dimensions, even on failure
        try:
            # Try to calculate features from the failed prompt
            with open(prompt_path, "r", encoding="utf-8") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except Exception:
            # Fallback values if prompt can't be read
            prompt_length, reasoning_sophistication = 0, 0.0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate_stage2(prompt_path):
    """
    Stage 2 evaluation: Comprehensive evaluation with adaptive sample size

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    global _EVALUATION_COUNTER, _MAX_ITERATIONS
    
    print("-" * 80)
    print("Starting Stage 2 evaluation...")
    print("-" * 80)

    try:
        # Загружаем max_iterations из конфига при первом вызове
        if _MAX_ITERATIONS == 100:
            try:
                with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
                    config_temp = yaml.safe_load(f)
                    _MAX_ITERATIONS = config_temp.get("max_iterations", 100)
            except Exception:
                pass  # Используем значение по умолчанию
        
        # Увеличиваем счетчик вызовов (если еще не был увеличен в Stage 1)
        # Примечание: OpenEvolve может вызывать Stage 2 отдельно, поэтому счетчик увеличиваем здесь тоже
        if _EVALUATION_COUNTER == 0:
            _EVALUATION_COUNTER = 1
        
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset (будет автоматически разделен для HotpotQA)
        dataset = load_hf_dataset(config)

        # Адаптивный размер выборки (используем текущий счетчик)
        stage2_samples = get_adaptive_sample_size(2, _EVALUATION_COUNTER, _MAX_ITERATIONS)
        
        print(f"Stage 2: Evaluating {stage2_samples} random samples (adaptive, iteration ~{_EVALUATION_COUNTER})...")

        # Run evaluation
        accuracy, correct, total = evaluate_prompt(prompt, dataset, config, stage2_samples)

        print(f"Stage 2 accuracy: {accuracy:.3f} ({correct}/{total})")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_sophistication = calculate_prompt_features(prompt)
        print(
            f"Prompt features - Length: {prompt_length} chars, Reasoning sophistication: {reasoning_sophistication:.3f}"
        )

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
        }

    except Exception as e:
        error_msg = str(e)
        # Краткое сообщение об ошибке без полного traceback
        print(f"Stage 2 evaluation failed: {error_msg}")
        print("-" * 80)

        # Always return feature dimensions, even on failure
        try:
            # Try to calculate features from the failed prompt
            with open(prompt_path, "r", encoding="utf-8") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_sophistication = calculate_prompt_features(failed_prompt)
        except Exception:
            # Fallback values if prompt can't be read
            prompt_length, reasoning_sophistication = 0, 0.0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_sophistication,
            "error": str(e),
        }


def evaluate(prompt_path):
    """
    Main evaluation function - for backwards compatibility
    Calls evaluate_stage2 for full evaluation

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    return evaluate_stage2(prompt_path)
