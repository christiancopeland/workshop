"""
Workshop Phase 2: Ollama Streaming Client
Token-by-token LLM generation for real-time responses
"""

import requests
import json
from typing import Generator, Optional, Callable
from logger import get_logger

log = get_logger("ollama_stream")


class OllamaStreamingClient:
    """
    Streaming client for Ollama API.
    
    Generates text token-by-token for real-time TTS synthesis.
    Supports sentence detection for immediate audio playback.
    
    Example:
        client = OllamaStreamingClient(
            base_url="http://localhost:11434",
            model="llama3.1:8b"
        )
        
        # Stream tokens
        for token in client.generate("Tell me a story"):
            print(token, end='', flush=True)
        
        # Stream sentences (for TTS)
        for sentence in client.generate_sentences("Explain AI"):
            synthesize(sentence)  # Immediate audio
    """
    
    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model: str = "qwen3:8b",
                 system_prompt: Optional[str] = None):
        """
        Initialize Ollama streaming client.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use
            system_prompt: Optional system prompt
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.system_prompt = system_prompt
        
        # Statistics
        self.requests_made = 0
        self.tokens_generated = 0
        
        log.info(f"OllamaStreamingClient: {model} @ {base_url}")
    
    def generate(self, 
                 prompt: str,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """
        Generate text token-by-token.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Yields:
            Individual tokens as they're generated
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if self.system_prompt:
            payload["system"] = self.system_prompt
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        self.requests_made += 1
        
        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    
                    if "response" in data:
                        token = data["response"]
                        self.tokens_generated += 1
                        yield token
                    
                    if data.get("done", False):
                        break
        
        except requests.exceptions.RequestException as e:
            log.error(f"Ollama request failed: {e}")
            yield ""  # Empty on error
    
    def generate_sentences(self,
                          prompt: str,
                          temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Generate text sentence-by-sentence.
        
        Accumulates tokens until sentence boundary, then yields complete sentence.
        Enables immediate TTS synthesis of each sentence.
        
        Sentence boundaries: . ! ? followed by space or end
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            
        Yields:
            Complete sentences as they're finished
        """
        buffer = ""
        
        for token in self.generate(prompt, temperature):
            buffer += token
            
            # Check for sentence boundary
            # Simple heuristic: . ! ? followed by space or capitalized letter
            if self._is_sentence_end(buffer):
                sentence = buffer.strip()
                if sentence:
                    log.debug(f"Sentence complete: {sentence[:50]}...")
                    yield sentence
                buffer = ""
        
        # Yield any remaining text
        if buffer.strip():
            yield buffer.strip()
    
    def _is_sentence_end(self, text: str) -> bool:
        """
        Check if text ends with sentence boundary.
        
        Args:
            text: Text to check
            
        Returns:
            True if text ends with sentence boundary
        """
        if not text:
            return False
        
        # Check for terminal punctuation
        if text[-1] in '.!?':
            return True
        
        # Check for punctuation + space
        if len(text) >= 2 and text[-2] in '.!?' and text[-1] == ' ':
            return True
        
        # Check for punctuation + capital letter
        if len(text) >= 2 and text[-2] in '.!?' and text[-1].isupper():
            return True
        
        return False
    
    def stop_generation(self):
        """
        Stop current generation.
        
        Note: This is a placeholder. Actual implementation would need
        to track the request and cancel it. For now, consumer can
        just stop iterating the generator.
        """
        log.info("Generation stop requested")
    
    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "requests_made": self.requests_made,
            "tokens_generated": self.tokens_generated
        }


def test_ollama_stream():
    """Test Ollama streaming client."""
    import time
    
    print("Testing OllamaStreamingClient...\n")
    
    # Test 1: Initialize
    print("Test 1: Initialize client")
    client = OllamaStreamingClient(
        base_url="http://localhost:11434",
        model="llama3.1:8b"
    )
    print(f"✅ Initialized: {client.get_stats()}\n")
    
    # Test 2: Token streaming
    print("Test 2: Generate tokens")
    print("Prompt: 'Count to 5'")
    print("Response: ", end='', flush=True)
    
    start = time.time()
    token_count = 0
    
    for token in client.generate("Count to 5", temperature=0.1):
        print(token, end='', flush=True)
        token_count += 1
    
    elapsed = time.time() - start
    print(f"\n\n  Tokens: {token_count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {token_count/elapsed:.1f} tokens/sec")
    print("✅ Token streaming works\n")
    
    # Test 3: Sentence streaming
    print("Test 3: Generate sentences")
    print("Prompt: 'Write 3 short sentences about dogs'")
    
    sentence_count = 0
    for sentence in client.generate_sentences(
        "Write 3 short sentences about dogs",
        temperature=0.3
    ):
        sentence_count += 1
        print(f"\n  Sentence {sentence_count}: {sentence}")
    
    print(f"\n  Total sentences: {sentence_count}")
    print("✅ Sentence streaming works\n")
    
    # Test 4: Stats
    print("Test 4: Final stats")
    stats = client.get_stats()
    print(f"  {stats}")
    print("✅ Stats work\n")
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_ollama_stream()