"""#454: with Remember settings on, an empty saved output folder reached the program as
--output-dir "", which resolved against the GUI backend's working folder (site-packages).
A blank output folder now counts as not given. The GUI is not started."""
import pytest

from whisperjav.webview_gui.api import WhisperJAVAPI

DEFAULT = r"C:\Users\someone\Documents\WhisperJAV\output"


@pytest.fixture
def api():
    a = WhisperJAVAPI.__new__(WhisperJAVAPI)
    a.default_output = DEFAULT
    return a


def output_dir_arg(args):
    i = args.index("--output-dir")
    return args[i + 1]


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_output_folder_is_not_passed_on(api, blank):
    assert api._output_dir({"output_dir": blank}) == DEFAULT
    assert api._output_dir({}) == DEFAULT


@pytest.mark.parametrize("given", ["source", r"D:\Subs", "SOURCE"])
def test_real_values_pass_through_unchanged(api, given):
    assert api._output_dir({"output_dir": given}) == given


@pytest.mark.parametrize("mode", ["balanced", "transformers", "qwen"])
def test_single_pass_command_never_gets_an_empty_output_dir(api, mode):
    args = api.build_args({"inputs": [r"D:\v\a.mp4"], "mode": mode, "output_dir": ""})
    assert output_dir_arg(args) == DEFAULT


def test_two_pass_command_never_gets_an_empty_output_dir(api):
    args = api._build_twopass_args({
        "inputs": [r"D:\v\a.mp4"],
        "output_dir": "  ",
        "pass1": {"pipeline": "balanced", "sensitivity": "balanced"},
        "pass2": {"pipeline": "qwen", "sensitivity": "balanced"},
    })
    assert output_dir_arg(args) == DEFAULT


def test_expected_output_paths_use_the_default_folder(api):
    paths = api._compute_expected_outputs({"inputs": [r"D:\v\a.mp4"], "mode": "balanced", "output_dir": ""})
    assert paths and all(p.startswith(DEFAULT) for p in paths)
