# Workshop Dashboard

Real-time monitoring dashboard for the Workshop AI agent system. Built with React, TypeScript, and Tailwind CSS.

## Quick Start

```bash
# Install dependencies
npm install

# Development server (with hot reload)
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

The dashboard connects to the Workshop WebSocket server at `ws://localhost:8766`.

## Tech Stack

| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| TypeScript | Type safety |
| Vite | Build tool & dev server |
| Zustand + Immer | State management |
| Tailwind CSS v3 | Styling |
| date-fns | Date formatting |

## Features

- **Real-time Event Stream**: Live feed of all agent events (tool calls, LLM requests, errors)
- **Task Tracking**: Monitor task progress with visual indicators
- **Research Library**: Browse and export research gathered by the agent
- **Subagent History**: Track spawned subagent executions
- **Memory Inspector**: Search semantic memory, view facts, projects, and user profile
- **Agent Status**: Live agent state, active model, and execution timer
- **Session Management**: Start, end, and resume sessions

## Development

For detailed architecture and code organization, see [LLM_CONTEXT.md](./LLM_CONTEXT.md).

## Build Output

Production builds are output to `dist/` and can be served by the Workshop backend via `dashboard.py`.
