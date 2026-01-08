import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { ResearchPlatform } from '../../types';
import { formatRelativeTime, cn } from '../../utils';

interface ResearchItemProps {
  platform: ResearchPlatform;
}

export function ResearchItem({ platform }: ResearchItemProps) {
  const { selectedResearch } = useDashboardStore();
  const { requestResearchDetail } = useWebSocket();

  const isSelected = selectedResearch?.filename === platform.filename;

  return (
    <div
      className={cn(
        'p-3 border-b border-border cursor-pointer transition-colors',
        isSelected ? 'bg-accent-blue/10 border-l-2 border-l-accent-blue' : 'hover:bg-bg-tertiary/50'
      )}
      onClick={() => requestResearchDetail(platform.filename)}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-text-primary truncate">
              {platform.topic}
            </span>
            {platform.is_active && (
              <span className="badge badge-green text-xs">Active</span>
            )}
          </div>
          <div className="text-xs text-text-secondary mt-1 line-clamp-2">
            {platform.original_query}
          </div>
        </div>
      </div>
      <div className="flex items-center justify-between mt-2 text-xs text-text-muted">
        <span>{platform.source_count} sources</span>
        <span>{platform.created_at ? formatRelativeTime(platform.created_at) : ''}</span>
      </div>
    </div>
  );
}
