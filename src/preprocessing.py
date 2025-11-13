"""
Preprocessing functions for data cleaning and transformation.
"""

import math
import re
import random
import ast

import nltk
from nltk.corpus import stopwords
import click

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


@click.group(help="Main CLI for data preprocessing tasks.")
def cli():
    """
    Main command-line interface (CLI) group for data preprocessing tasks.

    This is the top-level command group that organizes subcommands
    into categories for cleaning, numeric operations, text processing,
    and structural transformations.
    """


@cli.group(help="Commands related to data cleaning operations.")
def clean():
    """
    Group of commands for data cleaning operations.

    Includes commands for handling missing values in datasets,
    such as removing or filling missing entries.
    """


@clean.command(
    help="Remove missing values from a list."
    '\nExample: python script.py clean remove-missing --values \'[1, 2, None, 3, "", "nan"]\''
)
@click.option(
    "--values",
    required=True,
    help="List of values as a string, e.g., '[1, 2, None, 3]'",
)
def remove_missing(values):
    """
    Remove missing values from a list.

    Parameters
    ----------
    values : str
        A string representing a list of values, e.g., "[1, 2, None, 3]".

    Returns
    -------
    list
        List with missing values removed.

    Examples
    --------
    >>> python script.py clean remove-missing --values "[1, 2, None, 3, '', 'nan']"
    [1, 2, 3]
    """
    vals = ast.literal_eval(values)
    click.echo(remove_missing_values(vals))


@clean.command(
    help="Fill missing values in a list."
    "\nExample: python script.py clean fill-missing --values '[1, None, 2, \"\", 3]' --fill_value 0"
)
@click.option("--values", required=True, help="List of values as a string.")
@click.option(
    "--fill_value", default=0, help="Value used to fill missing entries (default=0)."
)
def fill_missing(values, fill_value):
    """
    Fill missing values in a list with a specified value.

    Parameters
    ----------
    values : str
        A string representing a list of values, e.g., "[1, None, '', 2]".
    fill_value : any, optional
        Value to replace missing entries with (default is 0).

    Returns
    -------
    list
        List with missing values replaced by the fill value.

    Examples
    --------
    >>> python script.py clean fill-missing --values "[1, None, '', 2]" --fill_value 0
    [1, 0, 0, 2]
    """
    vals = ast.literal_eval(values)
    click.echo(fill_missing_values(vals, fill_value))


@cli.group(help="Commands for numerical data transformations.")
def numeric():
    """
    Group of commands for numerical data transformations.

    Includes normalization, standardization, clipping,
    integer conversion, and logarithmic transformations.
    """


@numeric.command(
    help="Normalize numeric values using min-max scaling."
    "\nExample: python script.py numeric normalize --values '[1, 2, 3]' --new_min 0 --new_max 1"
)
@click.option("--values", required=True, help="List of numeric values as a string.")
@click.option("--new_min", default=0.0, help="New minimum value (default=0).")
@click.option("--new_max", default=1.0, help="New maximum value (default=1).")
def normalize(values, new_min, new_max):
    """
    Normalize numeric values using min-max scaling.

    Parameters
    ----------
    values : str
        A string representing a list of numeric values.
    new_min : float, optional
        Minimum value of the new range (default is 0).
    new_max : float, optional
        Maximum value of the new range (default is 1).

    Returns
    -------
    list
        Normalized list of numbers within the specified range.

    Examples
    --------
    >>> python script.py numeric normalize --values "[1, 2, 3]" --new_min 0 --new_max 1
    [0.0, 0.5, 1.0]
    """
    vals = ast.literal_eval(values)
    click.echo(min_max_normalize(vals, new_min, new_max))


@numeric.command(
    help="Standardize numeric values using z-score."
    "\nExample: python script.py numeric standardize --values '[1, 2, 3]'"
)
@click.option("--values", required=True, help="List of numeric values as a string.")
def standardize(values):
    """
    Standardize numeric values using z-score normalization.

    Parameters
    ----------
    values : str
        A string representing a list of numeric values.

    Returns
    -------
    list
        Z-score normalized values.

    Examples
    --------
    >>> python script.py numeric standardize --values "[1, 2, 3]"
    [-1.0, 0.0, 1.0]
    """
    vals = ast.literal_eval(values)
    click.echo(z_score_normalize(vals))


@numeric.command(
    help="Clip numeric values within a range."
    "\nExample: python script.py numeric clip --values '[1, 2, 10]' --min_value 0 --max_value 5"
)
@click.option("--values", required=True, help="List of numeric values as a string.")
@click.option("--min_value", default=0.0, help="Minimum allowed value (default=0).")
@click.option("--max_value", default=1.0, help="Maximum allowed value (default=1).")
def clip(values, min_value, max_value):
    """
    Clip numeric values to a specified range.

    Parameters
    ----------
    values : str
        A string representing a list of numeric values.
    min_value : float, optional
        Minimum allowable value (default is 0).
    max_value : float, optional
        Maximum allowable value (default is 1).

    Returns
    -------
    list
        List with clipped numeric values.

    Examples
    --------
    >>> python script.py numeric clip --values "[1, 5, 10]" --min_value 2 --max_value 6
    [2, 5, 6]
    """
    vals = ast.literal_eval(values)
    click.echo(clip_numerical_values(vals, min_value, max_value))


@numeric.command(
    help="Convert numeric values to integers."
    "\nExample: python script.py numeric to-int --values '[1.5, 2.9, 3.1]'"
)
@click.option("--values", required=True, help="List of numeric values as a string.")
def to_int(values):
    """
    Convert numeric values to integers.

    Parameters
    ----------
    values : str
        A string representing a list of numeric values.

    Returns
    -------
    list
        List of integer-converted values.

    Examples
    --------
    >>> python script.py numeric to-int --values "[1.5, 2.9, 3.1]"
    [1, 2, 3]
    """
    vals = ast.literal_eval(values)
    click.echo(to_integer_values(vals))


@numeric.command(
    help="Apply logarithmic transformation."
    "\nExample: python script.py numeric log --values '[1, 10, 100]'"
)
@click.option("--values", required=True, help="List of numeric values as a string.")
def log(values):
    """
    Apply logarithmic transformation to numeric values.

    Parameters
    ----------
    values : str
        A string representing a list of numeric values.

    Returns
    -------
    list
        Log-transformed numeric values.

    Examples
    --------
    >>> python script.py numeric log --values "[1, 10, 100]"
    [0.0, 2.302585092994046, 4.605170185988092]
    """
    vals = ast.literal_eval(values)
    click.echo(logarithmic_transform(vals))


@cli.group(help="Commands for text processing.")
def text():
    """
    Group of commands for text processing.

    Includes tokenization, punctuation removal,
    and stopword filtering.
    """


@text.command(
    help="Tokenize a string."
    "\nExample: python script.py text tokenize --text 'Hello world!'"
)
@click.option("--texts", required=True, help="Text string to tokenize.")
def tokenize(texts):
    """
    Tokenize a text string into lowercase word tokens.

    Parameters
    ----------
    texts : str
        Input text string.

    Returns
    -------
    list
        List of lowercase tokens.

    Examples
    --------
    >>> python script.py text tokenize --text "Hello world!"
    ['hello', 'world']
    """
    click.echo(tokenize_text(texts))


@text.command(
    help="Remove punctuation from text."
    "\nExample: python script.py text remove-punct --text 'Hello, world!'"
)
@click.option("--texts", required=True, help="Text string to process.")
def remove_punct(texts):
    """
    Remove punctuation from a text string, keeping only alphanumeric characters and spaces.

    Parameters
    ----------
    texts : str
        Input text string.

    Returns
    -------
    str
        Text string with punctuation removed.

    Examples
    --------
    >>> python script.py text remove-punct --text "Hello, world!"
    'Hello world'
    """
    click.echo(select_alphanumeric_and_spaces(texts))


@text.command(
    help="Remove stopwords from text."
    "\nExample: python script.py text remove-stopwords --text 'this is a simple example'"
)
@click.option("--texts", required=True, help="Text string to clean.")
def remove_stopwords_cmd(texts):
    """
    Remove stopwords from a text string using NLTK's English stopword list.

    Parameters
    ----------
    texts : str
        Input text string.

    Returns
    -------
    list
        List of tokens with stopwords removed.

    Examples
    --------
    >>> python script.py text remove-stopwords --text "this is a simple example"
    ['simple', 'example']
    """
    tokens = tokenize_text(texts)
    click.echo(remove_stopwords(tokens))


@cli.group(help="Commands related to data structure transformations.")
def struct():
    """
    Group of commands for structural data transformations.

    Includes operations for shuffling, flattening, and removing duplicates.
    """


@struct.command(
    help="Shuffle elements of a list."
    "\nExample: python script.py struct shuffle --values '[1, 2, 3, 4]'"
)
@click.option("--values", required=True, help="List of values as a string.")
def shuffle(values):
    """
    Randomly shuffle the elements of a list.

    Parameters
    ----------
    values : str
        A string representing a list of values.

    Returns
    -------
    list
        Shuffled list of values.

    Examples
    --------
    >>> python script.py struct shuffle --values "[1, 2, 3, 4]"
    [3, 1, 4, 2]
    """
    vals = ast.literal_eval(values)
    click.echo(random_shuffle_list(vals))


@struct.command(
    help="Flatten a nested list."
    "\nExample: python script.py struct flatten --values '[[1, 2], [3, [4, 5]]]'"
)
@click.option("--values", required=True, help="Nested list as a string.")
def flatten(values):
    """
    Flatten a nested list into a single list.

    Parameters
    ----------
    values : str
        A string representing a nested list.

    Returns
    -------
    list
        Flattened list containing all elements.

    Examples
    --------
    >>> python script.py struct flatten --values "[[1, 2], [3, [4, 5]]]"
    [1, 2, 3, 4, 5]
    """
    vals = ast.literal_eval(values)
    click.echo(flatten_list(vals))


@struct.command(
    help="Get unique values from a list."
    "\nExample: python script.py struct unique --values '[1, 2, 2, 3]'"
)
@click.option("--values", required=True, help="List of values as a string.")
def unique(values):
    """
    Retrieve unique values from a list while preserving order.

    Parameters
    ----------
    values : str
        A string representing a list of values.

    Returns
    -------
    list
        List containing unique values only.

    Examples
    --------
    >>> python script.py struct unique --values "[1, 2, 2, 3]"
    [1, 2, 3]
    """
    vals = ast.literal_eval(values)
    click.echo(remove_duplicates(vals))


if __name__ == "__main__":
    cli()
