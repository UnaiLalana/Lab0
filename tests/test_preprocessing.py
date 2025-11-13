import sys
import os
import pytest
from click.testing import CliRunner

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import preprocessing

@pytest.fixture
def sample_list():
    num_list = [1, 2, 3, 4, 5]
    return num_list


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_remove_missing_values():
    data = [1, None, '', 3.5, float('nan'), 'Hello', 0]
    cleaned = preprocessing.remove_missing_values(data)
    assert cleaned == [1, 3.5, 'Hello', 0]


def test_fill_missing_values():
    val = 42
    data = [1, None, '', 3.5, float('nan'), 'Hello', 0]
    cleaned = preprocessing.fill_missing_values(data, fill_value=val)
    assert cleaned == [1, val, val, 3.5, val, 'Hello', 0]


def test_remove_duplicates():
    data = [1, 2, 2, 3, 1, 4, 5, 3]
    unique = preprocessing.remove_duplicates(data)
    assert unique == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "data,expected",
    [
        ([10, 20, 30, 40, 50], [0.0, 0.25, 0.5, 0.75, 1.0]),
        ([5, 5, 5], [0.0, 0.0, 0.0]),
    ],
)
def test_min_max_normalize(data, expected):
    normalized = preprocessing.min_max_normalize(data)
    assert normalized == expected


def test_z_score_normalize(sample_list):
    normalized = preprocessing.z_score_normalize(sample_list)
    mean_result = sum(normalized) / len(normalized)
    assert abs(mean_result) < 1e-9


@pytest.mark.parametrize(
    "data,min_v,max_v,expected",
    [
        ([-10, 0, 5, 15, 25, 30], 0, 20, [0, 0, 5, 15, 20, 20]),
        ([1, 2, 3], 1, 3, [1, 2, 3]),
    ],
)
def test_clip_numerical_values(data, min_v, max_v, expected):
    clipped = preprocessing.clip_numerical_values(data, min_value=min_v, max_value=max_v)
    assert clipped == expected


def test_to_integer_values():
    data = [1.0, 2.5, 3.9, -4.2, '5', None]
    converted = preprocessing.to_integer_values(data)
    assert converted == [1, 2, 3, -4, 5]


@pytest.mark.parametrize(
    "data,expected",
    [
        ([1, 10, 100, 1000], [0.0, 1.0, 2.0, 3.0]),
        ([10, 100], [1.0, 2.0]),
    ],
)
def test_logarithmic_transform(data, expected):
    transformed = preprocessing.logarithmic_transform(data)
    assert transformed == expected


def test_tokenize_text():
    text = "Hello, world! --- This is a test."
    tokens = preprocessing.tokenize_text(text)
    assert tokens == ['hello', 'world', 'this', 'is', 'a', 'test']


def test_select_alphanumeric_and_spaces():
    text = "Hello, world! 123 @#$$%^&*()_+"
    cleaned = preprocessing.select_alphanumeric_and_spaces(text)
    assert cleaned == "Hello world 123 "


def test_remove_stopwords():
    text = "This is a simple test sentence for removing stopwords."
    cleaned = preprocessing.remove_stopwords(text.split(" "))
    assert cleaned == "This simple test sentence removing stopwords.".split(" ")


def test_flatten_list():
    nested = [1, [2, 3], [4, [5, 6]], 7]
    flat = preprocessing.flatten_list(nested)
    assert flat == [1, 2, 3, 4, 5, 6, 7]


def test_random_shuffle_list():
    data = [v for v in range(10000)]
    shuffled = preprocessing.random_shuffle_list(data)
    assert sorted(shuffled) == sorted(data)
    assert shuffled != data

def test_cli_remove_missing(cli_runner):
    result = cli_runner.invoke(
        preprocessing.cli,
        ["clean", "remove-missing", "--values", "[1, 2, None, '', 3]"]
    )
    assert result.exit_code == 0
    assert "1" in result.output
    assert "3" in result.output


def test_cli_fill_missing(cli_runner):
    result = cli_runner.invoke(
        preprocessing.cli,
        ["clean", "fill-missing", "--values", "[1, None, 2]", "--fill_value", "99"]
    )
    assert result.exit_code == 0
    assert "99" in result.output


def test_cli_normalize(cli_runner):
    result = cli_runner.invoke(
        preprocessing.cli,
        ["numeric", "normalize", "--values", "[1, 2, 3, 4]"]
    )
    assert result.exit_code == 0
    assert "0.0" in result.output


def test_cli_struct_shuffle(cli_runner):
    result = cli_runner.invoke(
        preprocessing.cli,
        ["struct", "shuffle", "--values", "[1, 2, 3, 4]"]
    )
    assert result.exit_code == 0
    assert any(str(i) in result.output for i in [1, 2, 3, 4])
