export type TaskStatus = 'pending' | 'in_progress' | 'completed';

export interface Task {
  id: string;
  content: string;
  status: TaskStatus;
  created_at: string;
  completed_at?: string;
}

export interface TaskState {
  has_tasks: boolean;
  tasks: Task[];
  original_request: string;
  stats: {
    total: number;
    completed: number;
    in_progress: number;
    pending: number;
  };
}

export const TASK_STATUS_ICONS: Record<TaskStatus, string> = {
  pending: '○',
  in_progress: '◐',
  completed: '●',
};

export const TASK_STATUS_COLORS: Record<TaskStatus, string> = {
  pending: 'text-text-muted',
  in_progress: 'text-accent-yellow',
  completed: 'text-accent-green',
};
