// Tool call record from subagent execution
export interface SubagentToolCall {
  call_id: string;
  tool_name: string;
  args: Record<string, unknown>;
  result: string;
  duration_ms: number;
  timestamp: string;
  success: boolean;
  error?: string;
}

// LLM call record from subagent execution
export interface SubagentLLMCall {
  call_id: string;
  iteration: number;
  model: string;
  messages_count: number;
  response_length: number;
  duration_ms: number;
  timestamp: string;
  prompt_tokens: number;
  completion_tokens: number;
  tool_calls_found: number;
  response_preview: string;
}

// Complete execution trace for a subagent
export interface SubagentTrace {
  trace_id: string;
  parent_trace_id?: string;
  started_at: string;
  completed_at: string;
  total_duration_ms: number;
  llm_call_count: number;
  tool_call_count: number;
  total_tokens: number;
  llm_calls: SubagentLLMCall[];
  tool_calls: SubagentToolCall[];
  system_prompt: string;
  context_injected: string;
  user_message: string;
  final_output: string;
  success: boolean;
  error_message?: string;
}

// Trace summary for list view
export interface TraceSummary {
  llm_call_count: number;
  tool_call_count: number;
  total_tokens: number;
  total_duration_ms: number;
}

// Shape returned by server from _get_subagent_history
export interface SubagentSnapshot {
  snapshot_id: string;
  timestamp: string;
  subagent_name: string;
  subagent_model: string;
  task: string;
  research_topic?: string;
  primary_model?: string;
  status: 'running' | 'completed' | 'failed';
  duration_ms?: number;
  output_length?: number;
  output_preview?: string;
  has_trace?: boolean;
  trace_summary?: TraceSummary;
}

// Full snapshot detail returned by get_snapshot_detail
export interface SubagentSnapshotDetail extends SubagentSnapshot {
  // Context that was provided to the subagent
  conversation_summary?: string;
  last_user_message?: string;
  pending_task?: string;
  telos_profile_summary?: string;
  telos_active_project?: string;
  telos_current_goals?: string[];
  research_platform_path?: string;
  research_source_count?: number;
  research_key_findings?: string;
  input_data?: Record<string, unknown>;
  output?: string;
  // Full execution trace
  trace?: SubagentTrace;
}

export interface SubagentHistory {
  snapshots: SubagentSnapshot[];
  active_count: number;
}
