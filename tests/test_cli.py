import os, sys
import pytest
from click.testing import CliRunner

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import preprocessing
from src import cli

@pytest.fixture
def cli_runner():
    return CliRunner()

def test_cli_remove_missing(cli_runner):
    result = cli_runner.invoke(
        cli.cli,
        ["clean", "remove-missing", "--values", "[1, 2, None, '', 3]"]
    )
    assert result.exit_code == 0
    assert "1" in result.output
    assert "3" in result.output


def test_cli_fill_missing(cli_runner):
    result = cli_runner.invoke(
        cli.cli,
        ["clean", "fill-missing", "--values", "[1, None, 2]", "--fill_value", "99"]
    )
    assert result.exit_code == 0
    assert "99" in result.output


def test_cli_normalize(cli_runner):
    result = cli_runner.invoke(
        cli.cli,
        ["numeric", "normalize", "--values", "[1, 2, 3, 4]"]
    )
    assert result.exit_code == 0
    assert "0.0" in result.output


def test_cli_struct_shuffle(cli_runner):
    result = cli_runner.invoke(
        cli.cli,
        ["struct", "shuffle", "--values", "[1, 2, 3, 4]"]
    )
    assert result.exit_code == 0
    assert any(str(i) in result.output for i in [1, 2, 3, 4])