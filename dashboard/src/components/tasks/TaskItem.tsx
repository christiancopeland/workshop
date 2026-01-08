import { Task, TASK_STATUS_ICONS, TASK_STATUS_COLORS } from '../../types';
import { cn } from '../../utils';

interface TaskItemProps {
  task: Task;
}

export function TaskItem({ task }: TaskItemProps) {
  const isCompleted = task.status === 'completed';
  const isInProgress = task.status === 'in_progress';

  return (
    <div className="task-item">
      {/* Status Icon */}
      <div
        className={cn(
          'text-lg shrink-0',
          TASK_STATUS_COLORS[task.status],
          isInProgress && 'animate-pulse'
        )}
      >
        {TASK_STATUS_ICONS[task.status]}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div
          className={cn(
            'text-sm',
            isCompleted ? 'text-text-muted line-through' : 'text-text-primary'
          )}
        >
          {task.content}
        </div>
        {isInProgress && (
          <div className="text-xs text-accent-yellow mt-1 flex items-center gap-1">
            <span className="w-1.5 h-1.5 bg-accent-yellow rounded-full animate-pulse" />
            Working on this...
          </div>
        )}
      </div>

      {/* Status Badge */}
      <div
        className={cn(
          'badge shrink-0',
          task.status === 'completed' && 'badge-green',
          task.status === 'in_progress' && 'badge-yellow',
          task.status === 'pending' && 'bg-bg-tertiary text-text-muted'
        )}
      >
        {task.status.replace('_', ' ')}
      </div>
    </div>
  );
}
