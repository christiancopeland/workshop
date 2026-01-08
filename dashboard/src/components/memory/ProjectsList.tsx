import { useDashboardStore } from '../../store';
import { cn } from '../../utils';

export function ProjectsList() {
  const { memoryProjects } = useDashboardStore();

  return (
    <div className="p-4">
      {memoryProjects.length === 0 ? (
        <div className="text-center text-text-muted py-8">
          No projects registered. Projects are tracked when you work on them.
        </div>
      ) : (
        <div className="space-y-2">
          {memoryProjects.map((project) => (
            <div
              key={project.path}
              className={cn(
                'panel p-3',
                project.is_active && 'border-accent-green/50 bg-accent-green/5'
              )}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg">📁</span>
                  <span className="text-sm font-medium text-text-primary">
                    {project.name}
                  </span>
                  {project.is_active && (
                    <span className="badge badge-green">Active</span>
                  )}
                </div>
              </div>
              <div className="text-xs text-text-muted mt-1 font-mono truncate">
                {project.path}
              </div>
              {project.last_accessed && (
                <div className="text-xs text-text-muted mt-1">
                  Last accessed: {project.last_accessed}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
