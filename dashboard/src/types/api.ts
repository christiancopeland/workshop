// Outbound actions (client -> server)
export type OutboundAction =
  | { action: 'send_message'; message: string }
  | { action: 'emergency_stop' }
  | { action: 'clear_history' }
  | { action: 'get_current_tasks' }
  | { action: 'get_research_library' }
  | { action: 'get_research_detail'; filename: string }
  | { action: 'get_subagent_history' }
  | { action: 'get_snapshot_detail'; snapshot_id: string }
  | { action: 'get_system_status' }
  | { action: 'get_memory_stats' }
  | { action: 'get_memory_facts' }
  | { action: 'get_memory_projects' }
  | { action: 'get_memory_profile' }
  | { action: 'get_memory_messages'; limit?: number }
  | { action: 'search_memory'; query: string; limit?: number }
  | { action: 'delete_memory_fact'; key: string }
  | { action: 'start_new_session'; mode?: string }
  | { action: 'end_session'; archive?: boolean }
  | { action: 'resume_session'; session_id: string }
  | { action: 'list_sessions'; limit?: number }
  | { action: 'clear_stale_state' };

// Inbound message types (server -> client)
export type InboundMessageType =
  | 'initial_state'
  | 'event'
  | 'task_list'
  | 'research_library'
  | 'research_detail'
  | 'subagent_history'
  | 'snapshot_detail'
  | 'system_status'
  | 'memory_stats'
  | 'memory_facts'
  | 'memory_projects'
  | 'memory_profile'
  | 'memory_messages'
  | 'memory_search_results'
  | 'session_info'
  | 'session_started'
  | 'session_ended'
  | 'session_resumed'
  | 'session_list'
  | 'stale_state'
  | 'agent_processing'
  | 'agent_idle'
  | 'emergency_stop_triggered'
  | 'error';

export interface InboundMessage {
  action?: InboundMessageType;  // Server sends 'action' field
  type?: InboundMessageType;    // Some events may use 'type'
  [key: string]: unknown;
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting';
