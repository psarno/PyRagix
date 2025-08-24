import json
import os
import multiprocessing
from typing import Dict, Any, Optional

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.formatted_text import HTML
    import config
except ImportError as e:
    print(f"Error: Required dependency not installed: {e}")
    print("Please install prompt_toolkit: pip install prompt_toolkit>=3.0.51")
    print("Or run: pip-compile requirements.in && pip install -r requirements.txt")
    raise SystemExit(1)


class ConfigMenu:
    def __init__(self, config_file: str = "user_config.json"):
        self.config_file = config_file
        self.param_types = {
            "TORCH_NUM_THREADS": int,
            "OPENBLAS_NUM_THREADS": int,
            "MKL_NUM_THREADS": int,
            "OMP_NUM_THREADS": int,
            "NUMEXPR_MAX_THREADS": int,
            "CUDA_VISIBLE_DEVICES": str,
            "PYTORCH_CUDA_ALLOC_CONF": str,
            "BATCH_SIZE": int,
            "INDEX_TYPE": str,
            "NLIST": int,
            "NPROBE": int,
            "SKIP_FILES": set,
            "BASE_DPI": int,
            "BATCH_SIZE_RETRY_DIVISOR": int,
            "INGESTION_LOG_FILE": str,
            "CRASH_LOG_FILE": str,
            "OLLAMA_BASE_URL": str,
            "OLLAMA_MODEL": str,
            "DEFAULT_TOP_K": int,
            "REQUEST_TIMEOUT": int,
            "TEMPERATURE": float,
            "TOP_P": float,
            "MAX_TOKENS": int,
        }
        self.valid_index_types = {"flat", "ivf_flat", "ivf_pq"}
        self.positive_int_keys = {
            "TORCH_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
            "BATCH_SIZE",
            "NLIST",
            "NPROBE",
            "BASE_DPI",
            "BATCH_SIZE_RETRY_DIVISOR",
            "DEFAULT_TOP_K",
            "REQUEST_TIMEOUT",
            "MAX_TOKENS",
        }
        # Load default params from config.py
        self.default_params = {key: getattr(config, key) for key in self.param_types}
        self.params = self.load_params()

    def load_params(self) -> Dict[str, Any]:
        """Load params from user_config.json or defaults from config.py."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    json_data = json.load(f)
                    # Convert SKIP_FILES list to set
                    if "SKIP_FILES" in json_data:
                        json_data["SKIP_FILES"] = set(json_data["SKIP_FILES"])
                    return json_data
            except json.JSONDecodeError:
                print(f"Error reading {self.config_file}. Using defaults.")
        return self.default_params.copy()

    def save_params(self) -> None:
        """Save params to user_config.json."""
        save_data = self.params.copy()
        # Convert set to list for JSON
        if "SKIP_FILES" in save_data:
            save_data["SKIP_FILES"] = list(save_data["SKIP_FILES"])
        with open(self.config_file, "w") as f:
            json.dump(save_data, f, indent=4)
        print(f"Config saved to {self.config_file}!")

    def validate_param(self, key: str, value: Any) -> bool:
        """Validate a single parameter (adapted from config.py)."""
        if key in self.positive_int_keys:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        elif key == "TEMPERATURE":
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 2.0):
                raise ValueError(f"{key} must be between 0.0 and 2.0")
        elif key == "TOP_P":
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                raise ValueError(f"{key} must be between 0.0 and 1.0")
        elif key in {
            "INGESTION_LOG_FILE",
            "CRASH_LOG_FILE",
            "OLLAMA_BASE_URL",
            "OLLAMA_MODEL",
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
        }:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
        elif key == "INDEX_TYPE":
            if value not in self.valid_index_types:
                raise ValueError(f"{key} must be one of {self.valid_index_types}")
        elif key == "SKIP_FILES":
            if not isinstance(value, set):
                raise ValueError(f"{key} must be a set")
        return True

    def get_display_value(self, value: Any) -> str:
        """Format value for display."""
        if isinstance(value, set):
            return ", ".join(sorted(value)) if value else "set()"
        return str(value)

    def parse_input(self, key: str, input_str: str) -> Any:
        """Parse input string to correct type."""
        if not input_str.strip():
            raise ValueError("Input cannot be empty")
        if self.param_types[key] == int:
            return int(input_str)
        elif self.param_types[key] == float:
            return float(input_str)
        elif self.param_types[key] == str:
            return input_str.strip()
        elif self.param_types[key] == set:
            return {s.strip() for s in input_str.split(",") if s.strip()}
        return input_str

    def suggest_auto_defaults(self) -> None:
        """Suggest sensible defaults based on system."""
        cpu_count = multiprocessing.cpu_count()
        suggested_threads = max(1, cpu_count // 2)
        print(
            f"Detected CPU cores: {cpu_count}. Suggesting {suggested_threads} for thread params."
        )

        cuda_available = False
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            suggested_cuda = "0" if cuda_available else ""
            print(
                f"CUDA available: {cuda_available}. Suggesting CUDA_VISIBLE_DEVICES: '{suggested_cuda}'"
            )
        except ImportError:
            print("Torch not installed; skipping CUDA check.")
            suggested_cuda = self.default_params["CUDA_VISIBLE_DEVICES"]

        confirm = input("Apply auto-suggestions? (y/n): ").lower().strip() == "y"
        if confirm:

            for key in [
                "TORCH_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMEXPR_MAX_THREADS",
            ]:
                self.params[key] = suggested_threads

            self.params["CUDA_VISIBLE_DEVICES"] = suggested_cuda

            print("Auto-suggestions applied!")

    def _create_menu_layout(self, keys: list, selected: int) -> Layout:
        """Create the layout for the interactive menu."""
        menu_lines = []
        menu_lines.append("=== RAG Pipeline Configuration Editor ===")
        menu_lines.append("")
        menu_lines.append(
            "Arrows: navigate | Enter: edit | s: save & exit | q: quit | a: auto-suggest"
        )
        menu_lines.append("")

        for i, key in enumerate(keys):
            if i == selected:
                display_val = self.get_display_value(self.params[key])
                menu_lines.append(f"<ansired>> {key}: {display_val}</ansired>")
            else:
                display_val = self.get_display_value(self.params[key])
                menu_lines.append(f"  {key}: {display_val}")

        text = "\n".join(menu_lines)
        return Layout(
            HSplit(
                [
                    Window(
                        content=FormattedTextControl(text=HTML(text)),
                        height=len(menu_lines) + 2,
                    )
                ]
            )
        )

    def run(self) -> Dict[str, Any]:
        """Run the interactive menu and return final params."""
        self.suggest_auto_defaults()  # Offer at start
        keys = list(self.params.keys())
        selected = 0
        app_running = True

        bindings = KeyBindings()

        @bindings.add(Keys.Up)
        def move_up(event):
            nonlocal selected
            selected = (selected - 1) % len(keys)

        @bindings.add(Keys.Down)
        def move_down(event):
            nonlocal selected
            selected = (selected + 1) % len(keys)

        @bindings.add(Keys.Enter)
        def edit_param(event):
            nonlocal app_running
            key = keys[selected]
            current_display = self.get_display_value(self.params[key])
            # Exit the app temporarily to get input
            event.app.exit()

            print(f"\nEditing {key} (current: {current_display})")
            if key == "SKIP_FILES":
                print("Enter comma-separated filenames to skip (or empty to clear):")
            try:
                new_input = input(f"New value for {key}: ").strip()
                if new_input:  # Only update if user provided input
                    new_value = self.parse_input(key, new_input)
                    self.validate_param(key, new_value)
                    self.params[key] = new_value
                    print(f"✓ {key} updated to: {self.get_display_value(new_value)}")
                else:
                    print("No change made (empty input).")
            except ValueError as e:
                print(f"Error: {e}. Value not updated.")

            input("\nPress Enter to continue...")

        @bindings.add("s")  # Save
        def save_and_exit(event):
            nonlocal app_running
            try:
                for key, value in self.params.items():
                    self.validate_param(key, value)
                self.save_params()
                app_running = False
                event.app.exit()
            except ValueError as e:
                event.app.exit()
                print(f"\nValidation failed: {e}. Please fix errors before saving.")
                input("Press Enter to continue...")

        @bindings.add("q")  # Quit
        def quit_menu(event):
            nonlocal app_running
            app_running = False
            event.app.exit()

        @bindings.add("a")  # Auto-suggest
        def auto_suggest(event):
            event.app.exit()
            self.suggest_auto_defaults()
            input("\nPress Enter to continue...")

        # Main menu loop
        while app_running:
            layout = self._create_menu_layout(keys, selected)
            app = Application(layout=layout, key_bindings=bindings, full_screen=False)

            try:
                app.run()
            except KeyboardInterrupt:
                print("\nExiting...")
                break

        return self.params


def launch_config_menu(
    config_file: str = "user_config.json",
) -> Optional[Dict[str, Any]]:
    """Launch the configuration menu and return the parameters."""
    try:
        menu = ConfigMenu(config_file)
        return menu.run()
    except KeyboardInterrupt:
        print("\n\nConfiguration cancelled by user.")
        return None
    except Exception as e:
        print(f"\nError running configuration menu: {e}")
        return None


if __name__ == "__main__":
    params = launch_config_menu()
    if params:
        print("\nFinal configuration saved!")
    else:
        print("\nConfiguration not saved.")
