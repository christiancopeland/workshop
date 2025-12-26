#!/usr/bin/env python3
"""
Workshop Phase 2: Automated System Fixes
Handles common configuration issues
"""

import subprocess
import sys
from pathlib import Path

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run_command(cmd, description, check=True):
    """Run shell command with feedback"""
    print(f"\n{BLUE}▶ {description}{RESET}")
    print(f"  Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        
        if result.returncode == 0:
            print(f"{GREEN}✅ Success{RESET}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"{RED}❌ Failed{RESET}")
            if result.stderr:
                print(result.stderr)
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"{RED}❌ Error: {e}{RESET}")
        return False

def fix_ollama():
    """Fix Ollama configuration"""
    print(f"\n{YELLOW}{'='*60}")
    print(f"FIX: OLLAMA")
    print(f"{'='*60}{RESET}\n")
    
    # Check if ollama is running
    result = subprocess.run(
        'pgrep -x ollama',
        shell=True,
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"{YELLOW}Ollama not running. Starting server...{RESET}")
        run_command(
            'ollama serve &',
            "Start Ollama server",
            check=False
        )
        print(f"\n{GREEN}Started Ollama server in background{RESET}")
        print(f"Give it 5 seconds to initialize...")
        import time
        time.sleep(5)
    else:
        print(f"{GREEN}Ollama already running{RESET}")
    
    # Check if model is pulled
    result = subprocess.run(
        'ollama list',
        shell=True,
        capture_output=True,
        text=True
    )
    
    if 'llama3.1' not in result.stdout:
        print(f"\n{YELLOW}Model llama3.1:8b not found. Pulling...{RESET}")
        run_command(
            'ollama pull llama3.1:8b',
            "Download llama3.1:8b model"
        )
    else:
        print(f"{GREEN}Model llama3.1:8b already available{RESET}")

def fix_piper():
    """Fix Piper voice model"""
    print(f"\n{YELLOW}{'='*60}")
    print(f"FIX: PIPER VOICE MODEL")
    print(f"{'='*60}{RESET}\n")
    
    # Check if piper is installed
    result = subprocess.run(
        'which piper',
        shell=True,
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"{YELLOW}Piper not found. Installing...{RESET}")
        run_command(
            'pip install piper-tts',
            "Install Piper TTS"
        )
    
    # Download voice model
    print(f"\n{YELLOW}Downloading Piper voice model...{RESET}")
    
    # Try different download methods
    success = run_command(
        'python -m piper.download_voices --locale en_US --quality medium',
        "Download en_US voice (medium quality)",
        check=False
    )
    
    if not success:
        # Alternative: Manual download
        print(f"\n{YELLOW}Trying alternative download method...{RESET}")
        
        voice_dir = Path.home() / '.local/share/piper-voices'
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Download directly from GitHub
        model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        
        run_command(
            f'wget -P {voice_dir} {model_url}',
            "Download voice model (.onnx)"
        )
        
        run_command(
            f'wget -P {voice_dir} {config_url}',
            "Download voice config (.json)"
        )
        
        print(f"\n{GREEN}Voice model downloaded to: {voice_dir}{RESET}")
        
        # Create symlink to expected location
        expected_dir = Path.home() / 'piper/voices'
        expected_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = voice_dir / 'en_US-lessac-medium.onnx'
        config_file = voice_dir / 'en_US-lessac-medium.onnx.json'
        
        if model_file.exists():
            link_model = expected_dir / 'en_US-lessac-medium.onnx'
            link_config = expected_dir / 'en_US-lessac-medium.onnx.json'
            
            if not link_model.exists():
                link_model.symlink_to(model_file)
            if not link_config.exists():
                link_config.symlink_to(config_file)
            
            print(f"{GREEN}Created symlinks at: {expected_dir}{RESET}")

def fix_openwakeword():
    """Install OpenWakeWord"""
    print(f"\n{YELLOW}{'='*60}")
    print(f"FIX: OPENWAKEWORD")
    print(f"{'='*60}{RESET}\n")
    
    run_command(
        'pip install openwakeword --break-system-packages',
        "Install OpenWakeWord"
    )

def update_config():
    """Update Workshop config with correct paths"""
    print(f"\n{YELLOW}{'='*60}")
    print(f"FIX: UPDATE WORKSHOP CONFIG")
    print(f"{'='*60}{RESET}\n")
    
    # Find actual Piper voice location
    voice_locations = [
        Path.home() / '.local/share/piper-voices/en_US-lessac-medium.onnx',
        Path.home() / 'piper/voices/en_US-lessac-medium.onnx',
    ]
    
    actual_voice = None
    for loc in voice_locations:
        if loc.exists():
            actual_voice = loc
            break
    
    if actual_voice:
        print(f"{GREEN}Found voice at: {actual_voice}{RESET}")
        print(f"\nUpdate piper_stream.py to use this path:")
        print(f"  model_path=\"{actual_voice}\"")
    else:
        print(f"{RED}Voice model not found after installation{RESET}")

def main():
    """Run all fixes"""
    print(f"\n{GREEN}{'='*60}")
    print(f"  WORKSHOP PHASE 2: AUTOMATED SYSTEM FIXES")
    print(f"{'='*60}{RESET}\n")
    
    print("This script will:")
    print("  1. Start Ollama server if not running")
    print("  2. Download LLM model if missing")
    print("  3. Download Piper voice model")
    print("  4. Install OpenWakeWord")
    print("  5. Update configurations")
    
    response = input(f"\n{YELLOW}Proceed? [y/N]: {RESET}").strip().lower()
    
    if response != 'y':
        print("Aborted.")
        return 1
    
    # Run fixes
    fix_ollama()
    fix_piper()
    fix_openwakeword()
    update_config()
    
    print(f"\n{GREEN}{'='*60}")
    print(f"  FIXES COMPLETE")
    print(f"{'='*60}{RESET}\n")
    
    print("Run system check again to verify:")
    print(f"  {BLUE}python system_check.py{RESET}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())