"""#444 (owner, 2026-10-06: custom servers keep their own temperature). Provider 'custom' with no explicit temperature:
the tone default is not sent and the request body carries no temperature. The end-to-end check against a stand-in
server is in the commit message; here the request-body wrap and the two flag sites."""
import inspect
import types

from pathlib import Path

from whisperjav.translate import core, service

CLI_SRC = (Path(core.__file__).parent / "cli.py").read_text(encoding="utf-8")  # importing it runs its parser


def test_patch_removes_temperature_from_the_request_body():
    def _generate_request_body(request, temperature):
        return {"temperature": temperature or 0.0, "model": "m", "messages": []}

    translator = types.SimpleNamespace(client=types.SimpleNamespace(_generate_request_body=_generate_request_body))
    assert core.apply_server_temperature_patch(translator)
    assert translator.client._generate_request_body(None, 0.8) == {"model": "m", "messages": []}


def test_patch_reports_when_the_client_has_no_request_builder():
    assert not core.apply_server_temperature_patch(types.SimpleNamespace(client=object()))


def test_both_paths_flag_only_custom_without_an_explicit_temperature():
    assert "if provider == 'custom' and temperature is None:" in inspect.getsource(service.translate_with_config)
    assert "if provider_name == 'custom' and getattr(args, 'temperature', None) is None:" in CLI_SRC
    src = inspect.getsource(core.translate_subtitle)
    assert "_server_temperature = provider_options.pop('_server_temperature', False)" in src
    assert "apply_server_temperature_patch(translator, debug=debug)" in src


def test_flags_are_popped_from_a_copy_so_every_file_of_a_run_keeps_them():
    """Review finding: the translate CLI passes ONE provider_options dict for all files of a run; popping the flags
    from it made files 2..N send temperature 0.0 (and lost _thinking_model)."""
    src = inspect.getsource(core.translate_subtitle)
    copy_at = src.index("provider_options = dict(provider_options) if provider_options else provider_options")
    assert copy_at < src.index("provider_options.pop('_thinking_model'")
    assert copy_at < src.index("provider_options.pop('_server_temperature'")
