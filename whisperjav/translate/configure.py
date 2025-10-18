"""
Interactive configuration wizard for whisperjav-translate.
"""

import os
import sys
from pathlib import Path

from .settings import load_settings, save_settings, get_settings_path, DEFAULT_SETTINGS
from .providers import PROVIDER_CONFIGS, SUPPORTED_TARGETS


def print_env_commands(provider: str, api_key: str, export_dotenv: bool = False):
    """Print environment variable commands for user's shell."""
    env_var = PROVIDER_CONFIGS[provider]['env_var']

    print("\nTo set the API key in your current session:")
    print("\n# For Windows Command Prompt:")
    print(f'set {env_var}={api_key}')
    print("\n# For Windows PowerShell:")
    print(f'$env:{env_var}="{api_key}"')
    print("\n# For Linux/Mac (bash/zsh):")
    print(f'export {env_var}="{api_key}"')

    if export_dotenv:
        print("\nTo persist the API key, add to your shell profile:")
        print(f"echo 'export {env_var}=\"{api_key}\"' >> ~/.bashrc  # or ~/.zshrc")


def write_dotenv_file(provider: str, api_key: str):
    """Write .env file for persistent API key storage."""
    dotenv_path = Path.cwd() / '.env'
    env_var = PROVIDER_CONFIGS[provider]['env_var']

    try:
        # Read existing .env if it exists
        existing = {}
        if dotenv_path.exists():
            with open(dotenv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing[key.strip()] = value.strip().strip('"\'')

        # Update with new API key
        existing[env_var] = api_key

        # Write back
        with open(dotenv_path, 'w', encoding='utf-8') as f:
            f.write("# WhisperJAV Translation API Keys\n")
            for key, value in existing.items():
                f.write(f'{key}="{value}"\n')

        print(f"\nAPI key saved to {dotenv_path}")
        print("To load it automatically, use: python-dotenv or similar")
        return True

    except Exception as e:
        print(f"\nWarning: Failed to write .env file: {e}")
        return False


def interactive_configure():
    """Run interactive configuration wizard."""
    print("="*60)
    print("WhisperJAV Translation Configuration Wizard")
    print("="*60)

    # Load existing settings
    settings = load_settings()
    print(f"\nSettings file: {get_settings_path()}")

    # Provider selection
    print("\n1. AI Provider")
    print("   Available providers:")
    for i, (key, config) in enumerate(PROVIDER_CONFIGS.items(), 1):
        current = " (current)" if settings.get('provider') == key else ""
        print(f"   {i}. {config['pysubtrans_name']} ({key}){current}")

    while True:
        choice = input(f"\n   Select provider [1-{len(PROVIDER_CONFIGS)}] (or Enter to keep current): ").strip()
        if not choice:
            provider = settings.get('provider', 'deepseek')
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(PROVIDER_CONFIGS):
                provider = list(PROVIDER_CONFIGS.keys())[idx]
                break
        except ValueError:
            pass
        print("   Invalid choice, try again")

    settings['provider'] = provider
    provider_config = PROVIDER_CONFIGS[provider]

    # API Key
    print(f"\n2. API Key for {provider_config['pysubtrans_name']}")
    print(f"   Environment variable: {provider_config['env_var']}")

    current_key = os.getenv(provider_config['env_var'], '')
    if current_key:
        print(f"   Current key from environment: {current_key[:8]}...")
        use_current = input("   Use current environment key? [Y/n]: ").strip().lower()
        if use_current != 'n':
            api_key = current_key
        else:
            api_key = input("   Enter API key: ").strip()
    else:
        api_key = input("   Enter API key: ").strip()

    if api_key:
        save_to_dotenv = input("   Save to .env file? [y/N]: ").strip().lower()
        if save_to_dotenv == 'y':
            write_dotenv_file(provider, api_key)
        else:
            print_env_commands(provider, api_key, export_dotenv=True)

    # Model (optional)
    print(f"\n3. Model (optional)")
    print(f"   Default model: {provider_config['model']}")
    model_override = input("   Override model? (Enter to use default): ").strip()
    if model_override:
        settings['model'] = model_override
    else:
        settings['model'] = None

    # Target language
    print(f"\n4. Target Language")
    print(f"   Available: {', '.join(SUPPORTED_TARGETS)}")
    current_target = settings.get('target_language', 'english')
    print(f"   Current: {current_target}")
    target = input(f"   Target language (Enter to keep {current_target}): ").strip().lower()
    if target and target in SUPPORTED_TARGETS:
        settings['target_language'] = target
    elif target:
        print(f"   Warning: Invalid target '{target}', keeping {current_target}")

    # Translation tone
    print(f"\n5. Translation Tone/Style")
    print("   1. standard (clean, professional)")
    print("   2. pornify (explicit, adult-oriented)")
    current_tone = settings.get('tone', 'standard')
    tone_choice = input(f"   Select tone [1-2] (Enter to keep {current_tone}): ").strip()
    if tone_choice == '1':
        settings['tone'] = 'standard'
    elif tone_choice == '2':
        settings['tone'] = 'pornify'

    # Advanced settings
    print(f"\n6. Advanced Settings (optional)")
    configure_advanced = input("   Configure advanced settings? [y/N]: ").strip().lower()
    if configure_advanced == 'y':
        # Scene threshold
        current_st = settings.get('scene_threshold', 60.0)
        st_input = input(f"   Scene threshold in seconds [{current_st}]: ").strip()
        if st_input:
            try:
                settings['scene_threshold'] = float(st_input)
            except ValueError:
                print(f"   Invalid value, keeping {current_st}")

        # Max batch size
        current_bs = settings.get('max_batch_size', 30)
        bs_input = input(f"   Max batch size [{current_bs}]: ").strip()
        if bs_input:
            try:
                settings['max_batch_size'] = int(bs_input)
            except ValueError:
                print(f"   Invalid value, keeping {current_bs}")

    # Save settings
    print("\n" + "="*60)
    print("Configuration Summary:")
    print("="*60)
    print(f"Provider: {provider_config['pysubtrans_name']} ({provider})")
    print(f"Model: {settings.get('model') or provider_config['model']}")
    print(f"Target Language: {settings['target_language']}")
    print(f"Tone: {settings['tone']}")
    print(f"Scene Threshold: {settings['scene_threshold']}s")
    print(f"Max Batch Size: {settings['max_batch_size']}")

    save = input("\nSave these settings? [Y/n]: ").strip().lower()
    if save != 'n':
        if save_settings(settings):
            print(f"\nSettings saved to: {get_settings_path()}")
            print("Configuration complete!")
        else:
            print("\nError saving settings")
            return False
    else:
        print("\nSettings not saved")

    return True


def configure_command():
    """Entry point for configure command."""
    try:
        interactive_configure()
    except KeyboardInterrupt:
        print("\n\nConfiguration cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during configuration: {e}")
        sys.exit(1)
