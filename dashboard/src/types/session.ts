export interface Session {
  id: string;
  started_at: string;
  ended_at?: string;
  message_count: number;
  task_count: number;
  mode?: string;
}

export interface StaleState {
  has_stale: boolean;
  task_count: number;
  last_activity: string;
  original_request?: string;
}

export interface SessionState {
  current: Session | null;
  history: Session[];
  stale: StaleState | null;
}
