export type EventType =
  | 'user_input'
  | 'assistant_response'
  | 'context_loading'
  | 'intent_detected'
  | 'skill_matched'
  | 'workflow_started'
  | 'tool_calling'
  | 'tool_result'
  | 'tool_error'
  | 'llm_calling'
  | 'llm_complete'
  | 'research_started'
  | 'research_query'
  | 'research_complete'
  | 'task_list_updated'
  | 'task_started'
  | 'task_completed'
  | 'subagent_spawned'
  | 'subagent_complete'
  | 'model_swap'
  | 'error'
  | 'warning'
  | 'info'
  | 'debug';

export interface DashboardEvent {
  id: string;
  type: EventType;
  timestamp: string;
  trace_id?: string;
  data: Record<string, unknown>;
}

export interface EventGroup {
  id: string;
  userInput: string;
  events: DashboardEvent[];
  stats: {
    tools: number;
    llm: number;
    errors: number;
    duration: number;
  };
  collapsed: boolean;
}

export type EventFilter = 'all' | 'tools' | 'llm' | 'errors' | 'grouped';

export interface EventStyle {
  icon: string;
  color: string;
  bgClass: string;
  borderClass: string;
}

export const EVENT_STYLES: Record<EventType, EventStyle> = {
  user_input: { icon: '👤', color: 'blue', bgClass: 'bg-accent-blue/10', borderClass: 'border-accent-blue/30' },
  assistant_response: { icon: '🤖', color: 'green', bgClass: 'bg-accent-green/10', borderClass: 'border-accent-green/30' },
  context_loading: { icon: '📂', color: 'purple', bgClass: 'bg-accent-purple/10', borderClass: 'border-accent-purple/30' },
  intent_detected: { icon: '🎯', color: 'cyan', bgClass: 'bg-accent-cyan/10', borderClass: 'border-accent-cyan/30' },
  skill_matched: { icon: '⚡', color: 'yellow', bgClass: 'bg-accent-yellow/10', borderClass: 'border-accent-yellow/30' },
  workflow_started: { icon: '🔄', color: 'purple', bgClass: 'bg-accent-purple/10', borderClass: 'border-accent-purple/30' },
  tool_calling: { icon: '🔧', color: 'yellow', bgClass: 'bg-accent-yellow/10', borderClass: 'border-accent-yellow/30' },
  tool_result: { icon: '✓', color: 'green', bgClass: 'bg-accent-green/10', borderClass: 'border-accent-green/30' },
  tool_error: { icon: '✗', color: 'red', bgClass: 'bg-accent-red/10', borderClass: 'border-accent-red/30' },
  llm_calling: { icon: '🧠', color: 'purple', bgClass: 'bg-accent-purple/10', borderClass: 'border-accent-purple/30' },
  llm_complete: { icon: '💭', color: 'purple', bgClass: 'bg-accent-purple/10', borderClass: 'border-accent-purple/30' },
  research_started: { icon: '🔬', color: 'cyan', bgClass: 'bg-accent-cyan/10', borderClass: 'border-accent-cyan/30' },
  research_query: { icon: '🔍', color: 'cyan', bgClass: 'bg-accent-cyan/10', borderClass: 'border-accent-cyan/30' },
  research_complete: { icon: '📊', color: 'green', bgClass: 'bg-accent-green/10', borderClass: 'border-accent-green/30' },
  task_list_updated: { icon: '📋', color: 'blue', bgClass: 'bg-accent-blue/10', borderClass: 'border-accent-blue/30' },
  task_started: { icon: '▶️', color: 'yellow', bgClass: 'bg-accent-yellow/10', borderClass: 'border-accent-yellow/30' },
  task_completed: { icon: '✅', color: 'green', bgClass: 'bg-accent-green/10', borderClass: 'border-accent-green/30' },
  subagent_spawned: { icon: '🚀', color: 'orange', bgClass: 'bg-accent-orange/10', borderClass: 'border-accent-orange/30' },
  subagent_complete: { icon: '🏁', color: 'green', bgClass: 'bg-accent-green/10', borderClass: 'border-accent-green/30' },
  model_swap: { icon: '🔀', color: 'purple', bgClass: 'bg-accent-purple/10', borderClass: 'border-accent-purple/30' },
  error: { icon: '❌', color: 'red', bgClass: 'bg-accent-red/10', borderClass: 'border-accent-red/30' },
  warning: { icon: '⚠️', color: 'yellow', bgClass: 'bg-accent-yellow/10', borderClass: 'border-accent-yellow/30' },
  info: { icon: 'ℹ️', color: 'blue', bgClass: 'bg-accent-blue/10', borderClass: 'border-accent-blue/30' },
  debug: { icon: '🐛', color: 'purple', bgClass: 'bg-accent-purple/10', borderClass: 'border-accent-purple/30' },
};
