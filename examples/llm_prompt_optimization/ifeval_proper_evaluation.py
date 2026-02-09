"""
Proper IFEval evaluation implementation.
This module provides constraint-based evaluation instead of simple length check.

The evaluation checks specific constraints from instructions such as:
- Exact sentence count requirements
- Word count limits
- Required words/phrases
- Forbidden words/characters
- Format requirements (lists, numbering, etc.)
"""

import re
from typing import Dict, List, Tuple, Optional
import json


def count_sentences(text: str) -> int:
    """Count sentences in text (split by . ! ?)."""
    # Simple sentence counting - can be improved
    sentences = re.split(r'[.!?]+', text.strip())
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def count_words(text: str) -> int:
    """Count words in text."""
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def check_exact_sentence_count(output: str, instruction: str) -> Optional[bool]:
    """Check if instruction requires exact sentence count."""
    # Look for patterns like "exactly N sentences", "N sentences", "write N sentences"
    patterns = [
        r'exactly\s+(\d+)\s+sentence',
        r'write\s+exactly\s+(\d+)\s+sentence',
        r'(\d+)\s+sentence[s]?\s+(?:only|exactly|precisely)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, instruction.lower())
        if match:
            required_count = int(match.group(1))
            actual_count = count_sentences(output)
            return actual_count == required_count
    
    return None  # No sentence count requirement found


def check_word_count(output: str, instruction: str) -> Optional[bool]:
    """Check if instruction requires specific word count."""
    # Look for patterns like "exactly N words", "at least N words", "at most N words"
    patterns = [
        (r'exactly\s+(\d+)\s+word', 'exact'),
        (r'write\s+exactly\s+(\d+)\s+word', 'exact'),
        (r'at\s+least\s+(\d+)\s+word', 'min'),
        (r'at\s+most\s+(\d+)\s+word', 'max'),
        (r'(\d+)\s+word[s]?\s+(?:only|exactly)', 'exact'),
    ]
    
    for pattern, constraint_type in patterns:
        match = re.search(pattern, instruction.lower())
        if match:
            required_count = int(match.group(1))
            actual_count = count_words(output)
            
            if constraint_type == 'exact':
                return actual_count == required_count
            elif constraint_type == 'min':
                return actual_count >= required_count
            elif constraint_type == 'max':
                return actual_count <= required_count
    
    return None  # No word count requirement found


def check_forbidden_words(output: str, instruction: str) -> Optional[bool]:
    """Check if output contains forbidden words."""
    # Look for patterns like "don't use X", "avoid X", "forbidden: X"
    forbidden_patterns = [
        r"don'?t\s+use\s+['\"]?([^'\".,!?]+)['\"]?",
        r"avoid\s+['\"]?([^'\".,!?]+)['\"]?",
        r"forbidden:\s*['\"]?([^'\".,!?]+)['\"]?",
        r"do\s+not\s+use\s+['\"]?([^'\".,!?]+)['\"]?",
        r"never\s+use\s+['\"]?([^'\".,!?]+)['\"]?",
    ]
    
    forbidden_words = []
    for pattern in forbidden_patterns:
        matches = re.findall(pattern, instruction.lower())
        forbidden_words.extend([w.strip() for m in matches for w in m.split(',')])
    
    if forbidden_words:
        output_lower = output.lower()
        for word in forbidden_words:
            if word.lower() in output_lower:
                return False  # Found forbidden word
        return True  # No forbidden words found
    
    return None  # No forbidden words requirement found


def check_forbidden_chars(output: str, instruction: str) -> Optional[bool]:
    """Check if output contains forbidden characters."""
    # Look for patterns like "don't use letter X", "without the letter X"
    patterns = [
        r"don'?t\s+use\s+(?:the\s+)?letter\s+['\"]?([^'\".,!?])['\"]?",
        r"without\s+(?:the\s+)?letter\s+['\"]?([^'\".,!?])['\"]?",
        r"avoid\s+(?:the\s+)?letter\s+['\"]?([^'\".,!?])['\"]?",
    ]
    
    forbidden_chars = []
    for pattern in patterns:
        matches = re.findall(pattern, instruction.lower())
        forbidden_chars.extend(matches)
    
    if forbidden_chars:
        output_lower = output.lower()
        for char in forbidden_chars:
            if char.lower() in output_lower:
                return False  # Found forbidden character
        return True  # No forbidden characters found
    
    return None  # No forbidden characters requirement found


def check_required_words(output: str, instruction: str) -> Optional[bool]:
    """Check if output contains required words."""
    # Look for patterns like "must include X", "use the word X"
    required_patterns = [
        r"must\s+include\s+['\"]?([^'\".,!?]+)['\"]?",
        r"use\s+(?:the\s+)?word\s+['\"]?([^'\".,!?]+)['\"]?",
        r"include\s+['\"]?([^'\".,!?]+)['\"]?",
    ]
    
    required_words = []
    for pattern in required_patterns:
        matches = re.findall(pattern, instruction.lower())
        required_words.extend([w.strip() for m in matches for w in m.split(',')])
    
    if required_words:
        output_lower = output.lower()
        for word in required_words:
            if word.lower() not in output_lower:
                return False  # Missing required word
        return True  # All required words found
    
    return None  # No required words requirement found


def check_list_format(output: str, instruction: str) -> Optional[bool]:
    """Check if instruction requires list format."""
    list_indicators = [
        'list', 'bullet', 'numbered', 'itemize', 'enumerate',
        '1.', '2.', '- ', '* ', '•'
    ]
    
    instruction_lower = instruction.lower()
    has_list_requirement = any(indicator in instruction_lower for indicator in list_indicators)
    
    if has_list_requirement:
        # Check if output has list-like structure
        has_numbered_list = bool(re.search(r'^\d+[.)]\s', output, re.MULTILINE))
        has_bullet_list = bool(re.search(r'^[-*•]\s', output, re.MULTILINE))
        return has_numbered_list or has_bullet_list
    
    return None  # No list format requirement found


def check_comma_requirement(output: str, instruction: str) -> Optional[bool]:
    """Check if instruction requires commas in sentences."""
    if 'comma' in instruction.lower():
        # Check if output has at least one comma
        return ',' in output
    
    return None  # No comma requirement found


def evaluate_ifeval_constraints(output: str, instruction: str) -> Tuple[bool, List[str]]:
    """
    Evaluate IFEval constraints comprehensively.
    
    Args:
        output: Model's output text
        instruction: The instruction that should be followed
        
    Returns:
        Tuple of (all_constraints_passed: bool, constraint_details: List[str])
    """
    if not output or len(output.strip()) < 5:
        return False, ["Output is too short or empty"]
    
    constraint_results = []
    all_passed = True
    
    # Check sentence count
    result = check_exact_sentence_count(output, instruction)
    if result is not None:
        status = "PASS" if result else "FAIL"
        constraint_results.append(f"Sentence count: {status}")
        if not result:
            all_passed = False
    
    # Check word count
    result = check_word_count(output, instruction)
    if result is not None:
        status = "PASS" if result else "FAIL"
        constraint_results.append(f"Word count: {status}")
        if not result:
            all_passed = False
    
    # Check forbidden words
    result = check_forbidden_words(output, instruction)
    if result is not None:
        status = "PASS" if result else "FAIL"
        constraint_results.append(f"Forbidden words: {status}")
        if not result:
            all_passed = False
    
    # Check forbidden characters
    result = check_forbidden_chars(output, instruction)
    if result is not None:
        status = "PASS" if result else "FAIL"
        constraint_results.append(f"Forbidden characters: {status}")
        if not result:
            all_passed = False
    
    # Check required words
    result = check_required_words(output, instruction)
    if result is not None:
        status = "PASS" if result else "FAIL"
        constraint_results.append(f"Required words: {status}")
        if not result:
            all_passed = False
    
    # Check list format
    result = check_list_format(output, instruction)
    if result is not None:
        status = "PASS" if result else "FAIL"
        constraint_results.append(f"List format: {status}")
        if not result:
            all_passed = False
    
    # Check comma requirement
    result = check_comma_requirement(output, instruction)
    if result is not None:
        status = "PASS" if result else "FAIL"
        constraint_results.append(f"Comma requirement: {status}")
        if not result:
            all_passed = False
    
    # If no specific constraints found, use basic validation
    if not constraint_results:
        # Fallback: at least check that output is non-trivial
        constraint_results.append("Basic check: Non-empty output")
        all_passed = len(output.strip()) > 10
    
    return all_passed, constraint_results


def evaluate_ifeval_example(output: str, instruction: str, verbose: bool = False) -> Dict:
    """
    Evaluate a single IFEval example with proper constraint checking.
    
    Args:
        output: Model's output text
        instruction: The instruction from IFEval dataset
        verbose: Whether to return detailed constraint information
        
    Returns:
        Dictionary with evaluation results
    """
    passed, details = evaluate_ifeval_constraints(output, instruction)
    
    result = {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
    }
    
    if verbose:
        result["constraint_details"] = details
        result["output"] = output
        result["instruction"] = instruction
    
    return result


if __name__ == "__main__":
    # Test the evaluation with example instructions
    test_cases = [
        {
            "instruction": "Write exactly two sentences explaining photosynthesis. Each sentence must have at least one comma.",
            "output": "Photosynthesis is the process by which plants convert light energy into chemical energy. This process occurs in chloroplasts, using carbon dioxide and water to produce glucose and oxygen.",
            "expected": True
        },
        {
            "instruction": "Write exactly two sentences explaining photosynthesis. Each sentence must have at least one comma.",
            "output": "Photosynthesis is important. Plants need it.",
            "expected": False  # No commas
        },
        {
            "instruction": "Write a paragraph about AI without using the letter 'e'.",
            "output": "Artificial intelligence is amazing.",
            "expected": False  # Contains 'e'
        },
        {
            "instruction": "Write a paragraph about AI without using the letter 'e'.",
            "output": "AI brings amazing things to our world. This technology impacts many domains.",
            "expected": False  # Contains 'e'
        },
    ]
    
    print("Testing IFEval constraint evaluation:")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        result = evaluate_ifeval_example(test["output"], test["instruction"], verbose=True)
        status = "✓" if result["passed"] == test["expected"] else "✗"
        print(f"\nTest {i}: {status}")
        print(f"Instruction: {test['instruction']}")
        print(f"Output: {test['output'][:100]}...")
        print(f"Passed: {result['passed']} (Expected: {test['expected']})")
        print(f"Details: {result['constraint_details']}")

