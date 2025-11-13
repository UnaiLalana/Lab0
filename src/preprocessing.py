"""
Preprocessing functions for data cleaning and transformation.
"""

import math
import re
import random

import nltk
from nltk.corpus import stopwords

if not nltk.download("stopwords", quiet=True):
    nltk.download("stopwords")


def remove_missing_values(values):
    """
    Removes missing values (None, '', nan) from a list.

    Args:
        values (list): A list that may contain missing values.

    Returns:
        list: A new list without missing values.
    """
    cleaned = []
    for v in values:
        if v is None or v == "" or isinstance(v, float) and math.isnan(v):
            continue
        cleaned.append(v)
    return cleaned


def fill_missing_values(values, fill_value=0):
    """
    Fills missing values (None, '', nan) in a list with a specified fill value.

    Args:
        values (list): A list that may contain missing values.
        fill_value: The value to replace missing values with (default 0).

    Returns:
        list: A new list with missing values filled.
    """
    filled = []
    for v in values:
        if v is None or v == "" or isinstance(v, float) and math.isnan(v):
            filled.append(fill_value)
        else:
            filled.append(v)
    return filled


def remove_duplicates(values):
    """
    Removes duplicate values from a list while preserving order.

    Args:
        values (list): A list that may contain duplicate values.

    Returns:
        list: A new list without duplicate values.
    """
    uniques = []
    for v in values:
        if v not in uniques:
            uniques.append(v)
    return uniques


def min_max_normalize(values, new_min=0, new_max=1):
    """
    Normalizes a list of numerical values to the range [0, 1] using min-max normalization.

    Args:
        values (list): A list of numerical values.

    Returns:
        list: A new list with normalized values.
    """
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [0.0 for _ in values]

    normalized = [
        ((v - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        for v in values
    ]
    return normalized


def z_score_normalize(values):
    """
    Normalizes a list of numerical values using z-score normalization.

    Args:
        values (list): A list of numerical values.

    Returns:
        list: A new list with z-score normalized values.
    """
    if not values:
        return []

    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return [0.0 for _ in values]

    normalized = [(v - mean_val) / std_dev for v in values]
    return normalized


def clip_numerical_values(values, min_value, max_value):
    """
    Clips numerical values in a list to be within a specified range [min_value, max_value].

    Args:
        values (list): A list of numerical values.
        min_value: The minimum allowable value.
        max_value: The maximum allowable value.

    Returns:
        list: A new list with clipped values.
    """
    clipped = [max(min(v, max_value), min_value) for v in values]
    return clipped


def to_integer_values(values):
    """
    Converts numerical values in a list to integers, ignoring non-convertible values.
    Args:
        values (list): A list of values.
    Returns:
        list: A new list with values converted to integers.
    """
    integers = []
    for v in values:
        try:
            num = int(v)
            integers.append(num)
        except (ValueError, TypeError):
            continue
    return integers


def logarithmic_transform(values):
    """
    Applies a logarithmic transformation to numerical values in a list.

    Args:
        values (list): A list of numerical values.

    Returns:
        list: A new list with logarithmically transformed values.
    """
    transformed = []
    for v in values:
        if v > 0:
            transformed.append(math.log10(v))
        else:
            continue
    return transformed


def tokenize_text(texts):
    """
    Tokenizes a string into words by splitting on whitespace.

    Args:
        text (str): The input string.

    Returns:
        list: A list of words.
    """
    texts = texts.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", texts)
    return tokens


def select_alphanumeric_and_spaces(texts):
    """
    Selects only alphanumeric characters and spaces from the input text.
    Args:
        text (str): The input string.
    Returns:
        str: The processed string containing only alphanumeric characters and spaces.
    """
    processed_text = re.sub(r"[^a-zA-Z0-9 ]+", "", texts)
    return processed_text


def remove_stopwords(tokens):
    """
    Removes common English stopwords from a list of tokens.

    Args:
        tokens (list): A list of word tokens.
    Returns:
        list: A new list with stopwords removed.
    """
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def flatten_list(nested_list):
    """
    Flattens a nested list into a single list.

    Args:
        nested_list (list): A list of lists.

    Returns:
        list: A flattened list.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def random_shuffle_list(values):
    """
    Randomly shuffles the elements of a list.

    Args:
        values (list): A list of values.

    Returns:
        list: A new list with elements randomly shuffled.
    """
    shuffled = values[:]
    random.shuffle(shuffled)
    return shuffled
