import { useEffect } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { SubagentItem } from './SubagentItem';
import { SubagentDetail } from './SubagentDetail';

export function SubagentsTab() {
  const { subagentHistory, selectedSubagent, setSelectedSubagent } = useDashboardStore();
  const { requestSubagentHistory } = useWebSocket();

  useEffect(() => {
    requestSubagentHistory();
  }, [requestSubagentHistory]);

  const runningCount = subagentHistory.filter((s) => s.status === 'running').length;
  const completedCount = subagentHistory.filter((s) => s.status === 'completed').length;
  const failedCount = subagentHistory.filter((s) => s.status === 'failed').length;

  const handleCloseDetail = () => {
    setSelectedSubagent(null);
  };

  return (
    <div className="h-full flex flex-col p-4">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-text-primary">
            🤖 Subagent History
          </h2>
          <button onClick={requestSubagentHistory} className="btn btn-ghost btn-sm">
            🔄 Refresh
          </button>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4 text-xs">
          <span className="text-text-muted">
            <span className="text-accent-yellow">●</span> {runningCount} running
          </span>
          <span className="text-text-muted">
            <span className="text-accent-green">●</span> {completedCount} completed
          </span>
          <span className="text-text-muted">
            <span className="text-accent-red">●</span> {failedCount} failed
          </span>
        </div>
      </div>

      {/* Subagent List */}
      <div className="flex-1 overflow-y-auto">
        {subagentHistory.length === 0 ? (
          <div className="flex items-center justify-center h-full text-text-muted">
            No subagent executions yet. Subagents are spawned for complex tasks.
          </div>
        ) : (
          <div className="space-y-3">
            {subagentHistory.map((snapshot, index) => (
              <SubagentItem key={snapshot.snapshot_id || index} snapshot={snapshot} />
            ))}
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedSubagent && (
        <SubagentDetail onClose={handleCloseDetail} />
      )}
    </div>
  );
}
