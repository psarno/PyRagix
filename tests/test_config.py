from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import pytest


class LoadIsolatedConfig(Protocol):
    """Protocol for the load_isolated_config fixture."""

    def __call__(self, tmp_path: Path, *, module_name: str) -> ModuleType: ...


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.py"
EXAMPLE_SETTINGS = PROJECT_ROOT / "settings.example.toml"


@pytest.fixture
def load_isolated_config() -> Generator[LoadIsolatedConfig, None, None]:
    """Load the config module in an isolated namespace using temporary settings."""

    loaded_modules: list[str] = []

    def _loader(tmp_path: Path, *, module_name: str) -> ModuleType:
        example_copy = tmp_path / "settings.example.toml"
        _ = shutil.copy(EXAMPLE_SETTINGS, example_copy)

        previous_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            spec = importlib.util.spec_from_file_location(module_name, CONFIG_PATH)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load module spec for {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded_modules.append(module_name)
            return module
        finally:
            os.chdir(previous_cwd)

    yield _loader

    for name in loaded_modules:
        _ = sys.modules.pop(name, None)


def test_settings_example_is_loadable(
    tmp_path: Path, load_isolated_config: LoadIsolatedConfig
) -> None:
    module = load_isolated_config(tmp_path, module_name="config_test_example")

    generated_settings = tmp_path / "settings.toml"
    example_settings = tmp_path / "settings.example.toml"

    assert generated_settings.exists(), "Auto-created settings.toml is missing"
    assert generated_settings.read_text(encoding="utf-8") == example_settings.read_text(
        encoding="utf-8"
    ), "settings.toml should match the tracked example"
    assert getattr(module, "INDEX_TYPE") == "ivf"


def test_index_type_normalization(
    tmp_path: Path, load_isolated_config: LoadIsolatedConfig
) -> None:
    module = load_isolated_config(tmp_path, module_name="config_test_normalization")
    config_cls: type[Any] = getattr(module, "PyRagixConfig")

    config_ivf = config_cls.model_validate({"INDEX_TYPE": "IVF"})
    config_ivf_flat = config_cls.model_validate({"INDEX_TYPE": "ivf_flat"})
    config_ivf_pq = config_cls.model_validate({"INDEX_TYPE": "IVF_PQ"})

    assert config_ivf.INDEX_TYPE == "ivf"
    assert config_ivf_flat.INDEX_TYPE == "ivf"
    assert config_ivf_pq.INDEX_TYPE == "ivf_pq"


def test_index_type_rejects_unknown_values(
    tmp_path: Path, load_isolated_config: LoadIsolatedConfig
) -> None:
    module = load_isolated_config(tmp_path, module_name="config_test_invalid")
    config_cls: type[Any] = getattr(module, "PyRagixConfig")

    with pytest.raises(ValueError):
        config_cls.model_validate({"INDEX_TYPE": "not-real"})


def test_missing_example_emits_warning(
    tmp_path: Path,
    load_isolated_config: LoadIsolatedConfig,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_isolated_config(tmp_path, module_name="config_test_missing_example")

    missing_example = tmp_path / "missing_settings.example.toml"
    target_settings = tmp_path / "generated_settings.toml"

    monkeypatch.setattr(module, "SETTINGS_EXAMPLE_FILE", str(missing_example))
    monkeypatch.setattr(module, "SETTINGS_FILE", str(target_settings))

    create_settings_func: Callable[[], None] = getattr(
        module, "_create_settings_from_example"
    )
    create_settings_func()

    captured = capsys.readouterr()
    assert str(missing_example) in captured.out
    assert "not found" in captured.out
    assert not target_settings.exists()
