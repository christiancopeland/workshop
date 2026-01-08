import { useEffect } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { TaskItem } from './TaskItem';

export function TasksTab() {
  const { tasks } = useDashboardStore();
  const { requestTasks } = useWebSocket();

  useEffect(() => {
    requestTasks();
  }, [requestTasks]);

  const progressPercent =
    tasks.stats.total > 0
      ? Math.round((tasks.stats.completed / tasks.stats.total) * 100)
      : 0;

  return (
    <div className="h-full flex flex-col p-4">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-text-primary">
            📋 Task Progress
          </h2>
          <button onClick={requestTasks} className="btn btn-ghost btn-sm">
            🔄 Refresh
          </button>
        </div>

        {/* Original Request */}
        {tasks.original_request && (
          <div className="bg-bg-secondary border border-border rounded-lg p-3 mb-4">
            <div className="text-xs text-text-muted mb-1">Original Request</div>
            <div className="text-sm text-text-primary">
              {tasks.original_request}
            </div>
          </div>
        )}

        {/* Progress Bar */}
        <div className="mb-2">
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="text-text-secondary">
              {tasks.stats.completed} of {tasks.stats.total} tasks completed
            </span>
            <span className="text-text-primary font-medium">{progressPercent}%</span>
          </div>
          <div className="progress-bar h-3">
            <div
              className="progress-fill"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4 text-xs">
          <span className="text-text-muted">
            <span className="text-accent-green">●</span> {tasks.stats.completed} completed
          </span>
          <span className="text-text-muted">
            <span className="text-accent-yellow">●</span> {tasks.stats.in_progress} in progress
          </span>
          <span className="text-text-muted">
            <span className="text-text-muted">●</span> {tasks.stats.pending} pending
          </span>
        </div>
      </div>

      {/* Task List */}
      <div className="flex-1 overflow-y-auto">
        {!tasks.has_tasks ? (
          <div className="flex items-center justify-center h-full text-text-muted">
            No active tasks. The agent will create tasks when needed.
          </div>
        ) : (
          <div className="panel">
            {tasks.tasks.map((task) => (
              <TaskItem key={task.id} task={task} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
