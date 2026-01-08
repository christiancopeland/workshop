export type AgentState =
  | 'idle'
  | 'routing'
  | 'thinking'
  | 'executing'
  | 'synthesizing'
  | 'error';

export interface AgentStatus {
  state: AgentState;
  details: string;
  model: string | null;
  timerStart: number | null;
}

export interface LoadedModel {
  name: string;
  size: number;
  sizeFormatted: string;
}

export interface SystemStatus {
  vram: {
    used: number;
    total: number;
    percent: number;
  } | null;
  loadedModels: LoadedModel[];
}

export const AGENT_STATE_LABELS: Record<AgentState, string> = {
  idle: 'Idle',
  routing: 'Routing...',
  thinking: 'Thinking...',
  executing: 'Executing...',
  synthesizing: 'Synthesizing...',
  error: 'Error',
};

export const AGENT_STATE_COLORS: Record<AgentState, string> = {
  idle: 'text-text-muted',
  routing: 'text-accent-purple',
  thinking: 'text-accent-blue',
  executing: 'text-accent-yellow',
  synthesizing: 'text-accent-cyan',
  error: 'text-accent-red',
};
