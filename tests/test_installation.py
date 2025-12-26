#!/usr/bin/env python3
"""
Quick test script to verify Workshop installation
Run this before trying the full system
"""

import sys


def test_python_version():
    """Check Python version"""
    print("1. Python version...", end=" ")
    if sys.version_info >= (3, 10):
        print(f"✅ {sys.version.split()[0]}")
        return True
    else:
        print(f"❌ {sys.version.split()[0]} (need 3.10+)")
        return False


def test_ollama():
    """Check if Ollama is running"""
    print("2. Ollama connection...", end=" ")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"✅ Found {len(models)} models")
            if model_names:
                print(f"   Available: {', '.join(model_names[:5])}")
            return True
        else:
            print(f"❌ Status {resp.status_code}")
            return False
    except ImportError:
        print("❌ requests not installed (pip install requests)")
        return False
    except Exception as e:
        print(f"❌ {e}")
        print("   Run: ollama serve")
        return False


def test_aiohttp():
    """Check aiohttp"""
    print("3. aiohttp...", end=" ")
    try:
        import aiohttp
        print(f"✅ {aiohttp.__version__}")
        return True
    except ImportError:
        print("❌ Not installed (pip install aiohttp)")
        return False


def test_chromadb():
    """Check ChromaDB"""
    print("4. ChromaDB...", end=" ")
    try:
        import chromadb
        print(f"✅ {chromadb.__version__}")
        return True
    except ImportError:
        print("⚠️  Not installed (pip install chromadb) - memory limited")
        return True  # Optional


def test_voice_deps():
    """Check voice dependencies"""
    print("5. Voice dependencies...")
    
    # Whisper
    print("   - Whisper...", end=" ")
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper")
    except ImportError:
        try:
            import whisper
            print("✅ openai-whisper")
        except ImportError:
            print("⚠️  Not installed (pip install faster-whisper)")
    
    # Audio
    print("   - Audio I/O...", end=" ")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ {len(devices)} devices")
    except ImportError:
        print("⚠️  Not installed (pip install sounddevice)")
    except Exception as e:
        print(f"⚠️  {e}")
    
    # TTS
    print("   - Piper TTS...", end=" ")
    try:
        import subprocess
        result = subprocess.run(["which", "piper"], capture_output=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.decode().strip()}")
        else:
            print("⚠️  Not found (optional)")
    except:
        print("⚠️  Not found (optional)")
    
    return True


def test_web_search():
    """Check web search"""
    print("6. Web search...", end=" ")
    try:
        from duckduckgo_search import DDGS
        print("✅ duckduckgo-search")
        return True
    except ImportError:
        print("⚠️  Not installed (pip install duckduckgo-search) - search disabled")
        return True  # Optional


def test_quick_ollama_response():
    """Test a quick Ollama response"""
    print("7. Ollama response test...", end=" ")
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "stream": False
            },
            timeout=30
        )
        if resp.status_code == 200:
            content = resp.json().get("message", {}).get("content", "")
            print(f"✅ Got response ({len(content)} chars)")
            return True
        else:
            print(f"❌ Status {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ {e}")
        return False


def main():
    print("=" * 50)
    print("Workshop Installation Test")
    print("=" * 50)
    print()
    
    results = []
    results.append(("Python", test_python_version()))
    results.append(("Ollama", test_ollama()))
    results.append(("aiohttp", test_aiohttp()))
    results.append(("ChromaDB", test_chromadb()))
    results.append(("Voice", test_voice_deps()))
    results.append(("Web Search", test_web_search()))
    results.append(("LLM Response", test_quick_ollama_response()))
    
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_pass = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed and name in ["Python", "Ollama", "aiohttp"]:
            all_pass = False
    
    print()
    if all_pass:
        print("🎉 Ready to go! Run: python main.py --mode text")
    else:
        print("⚠️  Fix the issues above before running Workshop")


if __name__ == "__main__":
    main()
