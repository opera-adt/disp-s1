from click.testing import CliRunner

from disp_s1.cli import cli_app

# https://click.palletsprojects.com/en/8.1.x/testing/


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("disp-s1, version ")
    assert result.output.endswith("\n")


def test_cli_debug():
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--debug"])
    assert result.exit_code == 2
    assert result.output.startswith("Usage: disp-s1 [OPTIONS] COMMAND [ARGS]...\n")
    assert result.output.endswith("\n")
