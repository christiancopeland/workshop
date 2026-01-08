import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { cn } from '../../utils';

export function Header() {
  const { connectionStatus, stats, currentSession } = useDashboardStore();
  const { emergencyStop, startNewSession, endSession } = useWebSocket();

  return (
    <header className="dashboard-header flex items-center justify-between px-4">
      {/* Left: Logo and Title */}
      <div className="flex items-center gap-3">
        <div className="text-2xl">🔧</div>
        <div>
          <h1 className="text-lg font-bold text-text-primary">Workshop Dashboard</h1>
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <span
              className={cn(
                'connection-dot',
                connectionStatus === 'connected' && 'connection-connected',
                connectionStatus === 'disconnected' && 'connection-disconnected',
                connectionStatus === 'reconnecting' && 'connection-reconnecting'
              )}
            />
            <span className="capitalize">{connectionStatus}</span>
            {currentSession?.id && (
              <>
                <span className="text-border">•</span>
                <span>Session: {currentSession.id.slice(0, 8)}</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Center: Stats */}
      <div className="flex items-center gap-4">
        <StatCard label="Messages" value={stats.messages} variant="blue" />
        <StatCard label="Tool Calls" value={stats.toolCalls} variant="yellow" />
        <StatCard label="LLM Calls" value={stats.llmCalls} variant="purple" />
        <StatCard label="Errors" value={stats.errors} variant="red" />
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        <SessionDropdown
          onNewSession={() => startNewSession()}
          onEndSession={() => endSession()}
        />
        <button
          onClick={emergencyStop}
          className="btn btn-danger btn-sm flex items-center gap-1"
        >
          <span>⏹</span>
          <span>Stop</span>
        </button>
      </div>
    </header>
  );
}

function StatCard({
  label,
  value,
  variant,
}: {
  label: string;
  value: number;
  variant: 'blue' | 'yellow' | 'purple' | 'red';
}) {
  const colorClass = {
    blue: 'text-accent-blue',
    yellow: 'text-accent-yellow',
    purple: 'text-accent-purple',
    red: 'text-accent-red',
  }[variant];

  return (
    <div className="stat-card min-w-[80px]">
      <div className={cn('stat-value', colorClass)}>{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function SessionDropdown({
  onNewSession,
  onEndSession,
}: {
  onNewSession: () => void;
  onEndSession: () => void;
}) {
  return (
    <div className="relative group">
      <button className="btn btn-ghost btn-sm flex items-center gap-1">
        <span>📋</span>
        <span>Session</span>
        <span className="text-xs">▼</span>
      </button>
      <div className="absolute right-0 top-full mt-1 bg-bg-secondary border border-border rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 min-w-[160px]">
        <button
          onClick={onNewSession}
          className="w-full px-4 py-2 text-left text-sm hover:bg-bg-tertiary flex items-center gap-2"
        >
          <span>➕</span>
          <span>New Session</span>
        </button>
        <button
          onClick={onEndSession}
          className="w-full px-4 py-2 text-left text-sm hover:bg-bg-tertiary flex items-center gap-2 text-accent-red"
        >
          <span>⏹</span>
          <span>End Session</span>
        </button>
      </div>
    </div>
  );
}
