"""v1.10.0 (N1): ConfigManager.save_config() writes its backup into the user settings folder
(<settings base>/WhisperJAV/config_backups/), never next to asr_config.json, which sits inside the
installed package. Backups written there were packed into the 1.9.x wheels."""
import shutil
from pathlib import Path

from whisperjav.config.manager import ConfigManager
from whisperjav.settings.gui_settings import get_gui_settings_path

PACKAGE_CONFIG = Path(__file__).resolve().parents[1] / "whisperjav" / "config" / "asr_config.json"


def test_backup_goes_to_user_settings_folder(tmp_path, monkeypatch):
    # Settings base: APPDATA on Windows, the home folder elsewhere.
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    package_dir = tmp_path / "package_config"
    package_dir.mkdir()
    config_path = package_dir / "asr_config.json"
    shutil.copy2(PACKAGE_CONFIG, config_path)

    manager = ConfigManager(config_path)
    manager.load_config()
    manager.update_ui_preference("n1_test", True)
    manager.save_config(backup=True)

    backup_dir = get_gui_settings_path().parent / "config_backups"
    assert str(backup_dir).startswith(str(tmp_path))
    backups = list(backup_dir.glob("asr_config.backup.*.json"))
    assert len(backups) == 1
    assert sorted(p.name for p in package_dir.iterdir()) == ["asr_config.json"]


def test_no_backup_when_not_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    config_path = tmp_path / "asr_config.json"
    shutil.copy2(PACKAGE_CONFIG, config_path)

    manager = ConfigManager(config_path)
    manager.load_config()
    manager.update_ui_preference("n1_test", True)
    manager.save_config(backup=False)

    assert not (get_gui_settings_path().parent / "config_backups").exists()
