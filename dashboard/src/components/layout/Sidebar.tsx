import { useDashboardStore } from '../../store';
import { formatBytes } from '../../utils';
import { AGENT_STATE_LABELS, AGENT_STATE_COLORS } from '../../types/agent';
import { cn } from '../../utils';
import { useEffect, useState } from 'react';

export function Sidebar() {
  return (
    <aside className="dashboard-sidebar p-4 flex flex-col gap-4">
      <ResourceMonitor />
      <AgentStatus />
      <ActiveTools />
      <RecentSkills />
    </aside>
  );
}

function ResourceMonitor() {
  const { vramUsage, loadedModels } = useDashboardStore();

  if (!vramUsage) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">💾 Resources</span>
        </div>
        <div className="panel-content text-text-muted text-sm">
          No VRAM data available
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">💾 Resources</span>
        <span className="text-xs text-text-muted">{vramUsage.percent.toFixed(1)}%</span>
      </div>
      <div className="panel-content space-y-3">
        <div>
          <div className="flex justify-between text-xs text-text-secondary mb-1">
            <span>VRAM Usage</span>
            <span>{formatBytes(vramUsage.used)} / {formatBytes(vramUsage.total)}</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${vramUsage.percent}%` }}
            />
          </div>
        </div>
        {loadedModels.length > 0 && (
          <div>
            <div className="text-xs text-text-muted mb-2">Loaded Models</div>
            <div className="space-y-1">
              {loadedModels.map((model) => (
                <div
                  key={model.name}
                  className="flex justify-between text-xs bg-bg-tertiary rounded px-2 py-1"
                >
                  <span className="text-text-secondary truncate">{model.name}</span>
                  <span className="text-text-muted">{model.sizeFormatted}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function AgentStatus() {
  const { agentState, agentDetails, agentModel, agentTimerStart } = useDashboardStore();
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!agentTimerStart) {
      setElapsed(0);
      return;
    }

    const interval = setInterval(() => {
      setElapsed(Date.now() - agentTimerStart);
    }, 100);

    return () => clearInterval(interval);
  }, [agentTimerStart]);

  const formatElapsed = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">🤖 Agent Status</span>
      </div>
      <div className="panel-content space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-text-secondary">State</span>
          <span className={cn('text-sm font-medium', AGENT_STATE_COLORS[agentState])}>
            {AGENT_STATE_LABELS[agentState]}
          </span>
        </div>
        {agentModel && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-secondary">Model</span>
            <span className="text-sm text-accent-purple">{agentModel}</span>
          </div>
        )}
        {agentTimerStart && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-secondary">Elapsed</span>
            <span className="text-sm text-text-primary font-mono">
              {formatElapsed(elapsed)}
            </span>
          </div>
        )}
        {agentDetails && (
          <div className="text-xs text-text-muted mt-2 bg-bg-tertiary rounded p-2">
            {agentDetails}
          </div>
        )}
      </div>
    </div>
  );
}

function ActiveTools() {
  const { activeTools } = useDashboardStore();

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">🔧 Active Tools</span>
        <span className="badge badge-yellow">{activeTools.length}</span>
      </div>
      <div className="panel-content">
        {activeTools.length === 0 ? (
          <div className="text-text-muted text-sm">No tools running</div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {activeTools.map((tool) => (
              <div
                key={tool}
                className="badge badge-yellow flex items-center gap-1"
              >
                <span className="w-2 h-2 bg-accent-yellow rounded-full animate-pulse" />
                {tool}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function RecentSkills() {
  const { recentSkills } = useDashboardStore();

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">⚡ Recent Skills</span>
      </div>
      <div className="panel-content">
        {recentSkills.length === 0 ? (
          <div className="text-text-muted text-sm">No skills used yet</div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {recentSkills.slice(0, 8).map((skill, i) => (
              <span key={`${skill}-${i}`} className="badge badge-purple">
                {skill}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
