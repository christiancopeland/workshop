# Workshop

Agentic voice assistant with context intelligence, powered by Claude Code CLI.

## Quick Start

```bash
# 1. Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | sh
claude login

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py --mode text
```

## Features

- **Claude Code as reasoning engine** - Superior reasoning via subscription (no per-token billing)
- **Two-stage semantic routing** - Embedding-based intent matching + Claude fallback
- **Skill-based tool execution** - 40+ tools across 6 skills
- **Personal context (Telos)** - Profile, goals, mission in `~/.workshop/Telos/`
- **Voice mode** - Wake word detection, TTS, real-time progress updates
- **WebSocket dashboard** - Real-time observability on port 8766

## Documentation

See [LLM_CONTEXT.md](LLM_CONTEXT.md) for comprehensive architecture documentation, including:
- Directory structure and key files
- Claude Code integration details
- Hook system and lifecycle events
- Skill and pattern systems
- Memory and session management
- Subagent orchestration
- Complete changelog

## License

MIT
