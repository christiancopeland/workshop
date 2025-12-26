#!/usr/bin/env python3
"""
Workshop Phase 2: Project-Aware System Status Check
Validates environment from current working directory
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
import socket

# Color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_section(title):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title.center(60)}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_status(component, status, details=""):
    if status == "OK":
        symbol = f"{GREEN}✅{RESET}"
    elif status == "WARN":
        symbol = f"{YELLOW}⚠️ {RESET}"
    else:
        symbol = f"{RED}❌{RESET}"
    print(f"{symbol} {component:<30} {details}")

def check_python_package(package_name):
    try:
        importlib.import_module(package_name)
        return True, ""
    except ImportError as e:
        return False, str(e)

def check_ollama_model(model_name):
    """Check if specific Ollama model exists"""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if model_name in result.stdout:
            return True, f"Model '{model_name}' available"
        return False, f"Model '{model_name}' not found"
    except Exception as e:
        return False, str(e)

def test_ollama_generation(model_name):
    """Actually test Ollama generation"""
    try:
        import requests
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': '1+1=',
                'stream': False,
                'options': {'num_predict': 5}
            },
            timeout=10
        )
        if response.status_code == 200:
            return True, "Generation test passed"
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_piper_synthesis(voice_path):
    """Actually test Piper synthesis"""
    try:
        import tempfile
        
        # Check for config file (both .json and .onnx.json)
        voice_path = Path(voice_path)
        config_path = voice_path.with_suffix('.onnx.json')
        if not config_path.exists():
            config_path = voice_path.with_suffix('.json')
        
        if not config_path.exists():
            return False, f"Config file not found: {config_path.name} or {voice_path.stem}.onnx.json"
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        result = subprocess.run(
            ['piper', '--model', str(voice_path), '--output_file', tmp_path],
            input=b'Test',
            capture_output=True,
            timeout=5
        )
        
        Path(tmp_path).unlink(missing_ok=True)
        
        if result.returncode == 0:
            return True, "Synthesis test passed"
        return False, result.stderr.decode()
    except Exception as e:
        return False, str(e)

def find_config_value(config_path, key):
    """Extract value from config.py"""
    try:
        with open(config_path) as f:
            for line in f:
                if line.strip().startswith(key):
                    # Extract value after '='
                    value = line.split('=', 1)[1].strip()
                    
                    # Handle os.getenv() pattern
                    if 'os.getenv' in value or 'getenv' in value:
                        # Extract default value from getenv("VAR", "default")
                        import re
                        match = re.search(r'getenv\([^,]+,\s*["\']([^"\']+)["\']', value)
                        if match:
                            value = match.group(1)
                        else:
                            # Try without quotes
                            match = re.search(r'getenv\([^,]+,\s*([^)]+)\)', value)
                            if match:
                                value = match.group(1).strip()
                    
                    # Remove quotes
                    value = value.strip('"\'')
                    return value
    except Exception:
        pass
    return None

def main():
    print(f"\n{GREEN}{'='*60}")
    print(f"  WORKSHOP PHASE 2: PROJECT-AWARE SYSTEM CHECK")
    print(f"{'='*60}{RESET}\n")
    
    # Get current directory
    cwd = Path.cwd()
    print(f"Working directory: {cwd}\n")
    
    issues = []
    warnings = []
    
    # ==================== CONFIG FILE ====================
    print_section("CONFIGURATION")
    
    config_path = cwd / 'config.py'
    if config_path.exists():
        print_status("config.py", "OK", str(config_path))
        
        # Extract model name from config
        model_name = find_config_value(config_path, 'MODEL')
        if model_name:
            print(f"  Configured model: {model_name}")
        else:
            model_name = "llama3.1:8b"  # fallback
            warnings.append("MODEL not found in config.py, using default")
    else:
        print_status("config.py", "ERROR", "Not found")
        issues.append("Missing config.py in project directory")
        model_name = "llama3.1:8b"
    
    # ==================== PYTHON PACKAGES ====================
    print_section("PYTHON ENVIRONMENT")
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"Path: {sys.executable}\n")
    
    packages = {
        'numpy': 'Core numerical library',
        'sounddevice': 'Audio I/O',
        'requests': 'HTTP client',
        'openwakeword': 'Wake word detection',
    }
    
    for pkg, desc in packages.items():
        ok, msg = check_python_package(pkg)
        status = "OK" if ok else "ERROR"
        print_status(pkg, status, desc)
        if not ok:
            issues.append(f"Install {pkg}: pip install {pkg}")
    
    # ==================== OLLAMA ====================
    print_section("OLLAMA")
    
    # Check server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        
        if result == 0:
            print_status("Server", "OK", "Running on port 11434")
        else:
            print_status("Server", "ERROR", "Not responding")
            issues.append("Start Ollama: ollama serve")
    except Exception as e:
        print_status("Server", "ERROR", str(e))
        issues.append("Start Ollama server")
    
    # Check model
    ok, msg = check_ollama_model(model_name)
    print_status(f"Model ({model_name})", "OK" if ok else "WARN", msg)
    if not ok:
        warnings.append(f"Pull model: ollama pull {model_name}")
    
    # Test generation
    ok, msg = test_ollama_generation(model_name)
    print_status("Generation test", "OK" if ok else "ERROR", msg)
    if not ok:
        issues.append("Ollama generation failed - check server and model")
    
    # ==================== WHISPER.CPP ====================
    print_section("WHISPER.CPP")
    
    whisper_dir = Path.home() / "whisper.cpp"
    whisper_bin = whisper_dir / "build/bin/whisper-cli"
    whisper_model = whisper_dir / "models/ggml-base.en.bin"
    
    for item, path in [
        ("Directory", whisper_dir),
        ("Binary", whisper_bin),
        ("Model", whisper_model)
    ]:
        if path.exists():
            print_status(item, "OK", str(path))
        else:
            print_status(item, "ERROR", "Not found")
            issues.append(f"Whisper.cpp {item.lower()} missing")
    
    # ==================== PIPER ====================
    print_section("PIPER VOICES")
    
    # Check project directory first
    voice_files = list(cwd.glob("*.onnx"))
    
    if voice_files:
        voice_path = voice_files[0]
        print_status("Voice model", "OK", voice_path.name)
        
        # Check for config file
        config_path = voice_path.with_suffix('.onnx.json')
        alt_config = voice_path.with_suffix('.json')
        
        if config_path.exists():
            print_status("Voice config", "OK", config_path.name)
        elif alt_config.exists():
            print_status("Voice config", "WARN", f"{alt_config.name} (should be .onnx.json)")
            warnings.append(f"Rename config: mv {alt_config.name} {config_path.name}")
        else:
            print_status("Voice config", "ERROR", "Not found")
            issues.append("Piper requires .onnx.json config file alongside model")
        
        # Test synthesis
        ok, msg = test_piper_synthesis(str(voice_path))
        print_status("Synthesis test", "OK" if ok else "ERROR", msg)
        if not ok:
            issues.append("Piper synthesis failed - check voice model and config")
    else:
        # Check standard locations
        standard_locs = [
            Path.home() / ".local/share/piper-voices",
            Path.home() / ".local/share/piper/voices",
            Path.home() / "piper/voices"
        ]
        
        found = False
        for loc in standard_locs:
            voice_files = list(loc.glob("*.onnx"))
            if voice_files:
                print_status("Voice model", "OK", str(voice_files[0]))
                found = True
                break
        
        if not found:
            print_status("Voice model", "ERROR", "Not found")
            issues.append("Download Piper voice: python -m piper.download_voices --locale en_US")
    
    # ==================== WORKSHOP COMPONENTS ====================
    print_section("WORKSHOP COMPONENTS")
    
    components = [
        'audio_realtime.py',
        'audio_playback.py', 
        'voice_state.py',
        'whisper_wrapper.py',
        'wake_word.py',
        'ollama_stream.py',
        'piper_stream.py',
    ]
    
    for comp in components:
        path = cwd / comp
        if path.exists():
            print_status(comp, "OK", f"{path.stat().st_size} bytes")
        else:
            print_status(comp, "ERROR", "Missing")
            issues.append(f"Component missing: {comp}")
    
    # ==================== AUDIO DEVICES ====================
    print_section("AUDIO DEVICES")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        # Find Blue Microphones device
        blue_mic = None
        for i, device in enumerate(devices):
            if 'Blue Microphones' in device['name']:
                blue_mic = (i, device['name'])
                break
        
        if blue_mic:
            print_status("Blue Microphone", "OK", f"Device {blue_mic[0]}: {blue_mic[1]}")
        else:
            print_status("Blue Microphone", "WARN", "Not found (will use default)")
            warnings.append("Blue Microphones USB not detected")
        
        default_input = sd.query_devices(kind='input')
        print_status("Default Input", "OK", default_input['name'])
        
        default_output = sd.query_devices(kind='output')
        print_status("Default Output", "OK", default_output['name'])
    
    except Exception as e:
        print_status("Audio check", "ERROR", str(e))
    
    # ==================== SUMMARY ====================
    print_section("SUMMARY")
    
    if not issues and not warnings:
        print(f"{GREEN}✅ ALL SYSTEMS GO!{RESET}\n")
        print("Your Workshop Phase 2 environment is fully configured.")
        print("Ready to proceed with integration testing.\n")
        return 0
    
    if warnings:
        print(f"{YELLOW}⚠️  WARNINGS ({len(warnings)}){RESET}")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print()
    
    if issues:
        print(f"{RED}❌ CRITICAL ISSUES ({len(issues)}){RESET}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print(f"{YELLOW}Fix critical issues before proceeding.{RESET}\n")
        return 1
    
    print(f"{YELLOW}System functional with warnings.{RESET}")
    print("Can proceed with caution.\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())