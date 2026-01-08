import { SubagentSnapshot } from '../../types';
import { formatDuration, formatRelativeTime, cn } from '../../utils';
import { useWebSocket } from '../../hooks/useWebSocket';

interface SubagentItemProps {
  snapshot: SubagentSnapshot;
}

export function SubagentItem({ snapshot }: SubagentItemProps) {
  const { requestSnapshotDetail } = useWebSocket();

  const statusColor = {
    running: 'badge-yellow',
    completed: 'badge-green',
    failed: 'badge-red',
  }[snapshot.status] || 'badge-gray';

  // Safely format timestamp - handle invalid dates
  const formatTimestamp = (ts: string | undefined) => {
    if (!ts) return '';
    try {
      const date = new Date(ts);
      if (isNaN(date.getTime())) return '';
      return formatRelativeTime(ts);
    } catch {
      return '';
    }
  };

  const handleClick = () => {
    requestSnapshotDetail(snapshot.snapshot_id);
  };

  return (
    <div className="panel">
      <div
        className="p-3 cursor-pointer hover:bg-bg-tertiary/50 transition-colors"
        onClick={handleClick}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className={cn('badge', statusColor)}>
                {snapshot.status}
              </span>
              <span className="badge badge-purple">{snapshot.subagent_model || snapshot.primary_model || 'unknown'}</span>
              {snapshot.has_trace && (
                <span className="badge badge-blue" title="Has execution trace">
                  📊 Trace
                </span>
              )}
            </div>
            <div className="text-sm text-text-primary">{snapshot.task || snapshot.research_topic || 'No task description'}</div>
            {snapshot.subagent_name && (
              <div className="text-xs text-text-muted mt-1">Agent: {snapshot.subagent_name}</div>
            )}
          </div>
          <div className="text-right shrink-0">
            {snapshot.duration_ms && (
              <div className="text-sm text-text-secondary font-mono">
                {formatDuration(snapshot.duration_ms)}
              </div>
            )}
            <div className="text-xs text-text-muted">
              {formatTimestamp(snapshot.timestamp)}
            </div>
          </div>
        </div>

        {/* Trace summary */}
        {snapshot.trace_summary && (
          <div className="flex items-center gap-3 mt-2 text-xs text-text-muted">
            <span title="LLM Calls">🤖 {snapshot.trace_summary.llm_call_count}</span>
            <span title="Tool Calls">🔧 {snapshot.trace_summary.tool_call_count}</span>
            <span title="Total Tokens">📝 {snapshot.trace_summary.total_tokens}</span>
          </div>
        )}

        {/* Click hint */}
        <div className="text-xs text-text-muted mt-2 flex items-center gap-1">
          <span>▶</span>
          <span>Click to view details</span>
          {snapshot.output_length ? ` (${snapshot.output_length} chars output)` : ''}
        </div>
      </div>
    </div>
  );
}
