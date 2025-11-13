"""
Command-line interface (CLI) for data preprocessing tasks.
"""

import ast
import click
from .preprocessing import (
    remove_missing_values,
    fill_missing_values,
    remove_duplicates,
    min_max_normalize,
    z_score_normalize,
    clip_numerical_values,
    to_integer_values,
    logarithmic_transform,
    tokenize_text,
    select_alphanumeric_and_spaces,
    remove_stopwords,
    flatten_list,
    random_shuffle_list,
)


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
