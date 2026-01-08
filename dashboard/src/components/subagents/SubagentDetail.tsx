import { useDashboardStore } from '../../store';
import { formatDuration, cn } from '../../utils';
import type { SubagentSnapshotDetail, SubagentLLMCall, SubagentToolCall } from '../../types';

interface SubagentDetailProps {
  onClose: () => void;
}

function LLMCallItem({ call, index }: { call: SubagentLLMCall; index: number }) {
  return (
    <div className="panel p-3 mb-2">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="badge badge-blue">LLM Call #{index + 1}</span>
          <span className="text-xs text-text-muted">Iteration {call.iteration + 1}</span>
        </div>
        <div className="text-xs text-text-muted font-mono">
          {formatDuration(call.duration_ms)}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs mb-2">
        <div>
          <span className="text-text-muted">Model:</span>{' '}
          <span className="text-text-secondary">{call.model}</span>
        </div>
        <div>
          <span className="text-text-muted">Messages:</span>{' '}
          <span className="text-text-secondary">{call.messages_count}</span>
        </div>
        <div>
          <span className="text-text-muted">Tokens:</span>{' '}
          <span className="text-text-secondary">
            {call.prompt_tokens} prompt + {call.completion_tokens} completion
          </span>
        </div>
        <div>
          <span className="text-text-muted">Tool Calls:</span>{' '}
          <span className="text-text-secondary">{call.tool_calls_found}</span>
        </div>
      </div>
      {call.response_preview && (
        <div className="mt-2">
          <div className="text-xs text-text-muted mb-1">Response Preview:</div>
          <pre className="text-xs text-text-secondary bg-bg-tertiary p-2 rounded overflow-x-auto max-h-32 overflow-y-auto whitespace-pre-wrap">
            {call.response_preview}
          </pre>
        </div>
      )}
    </div>
  );
}

function ToolCallItem({ call, index }: { call: SubagentToolCall; index: number }) {
  return (
    <div className="panel p-3 mb-2">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={cn('badge', call.success ? 'badge-green' : 'badge-red')}>
            {call.tool_name}
          </span>
          <span className="text-xs text-text-muted">#{index + 1}</span>
        </div>
        <div className="text-xs text-text-muted font-mono">
          {formatDuration(call.duration_ms)}
        </div>
      </div>
      {Object.keys(call.args).length > 0 && (
        <div className="mb-2">
          <div className="text-xs text-text-muted mb-1">Arguments:</div>
          <pre className="text-xs text-text-secondary bg-bg-tertiary p-2 rounded overflow-x-auto">
            {JSON.stringify(call.args, null, 2)}
          </pre>
        </div>
      )}
      <div>
        <div className="text-xs text-text-muted mb-1">Result:</div>
        <pre className="text-xs text-text-secondary bg-bg-tertiary p-2 rounded overflow-x-auto max-h-40 overflow-y-auto whitespace-pre-wrap">
          {call.result || '(empty)'}
        </pre>
      </div>
      {call.error && (
        <div className="mt-2 text-xs text-accent-red">
          Error: {call.error}
        </div>
      )}
    </div>
  );
}

export function SubagentDetail({ onClose }: SubagentDetailProps) {
  const { selectedSubagent } = useDashboardStore();

  if (!selectedSubagent) {
    return null;
  }

  const snapshot = selectedSubagent as SubagentSnapshotDetail;
  const trace = snapshot.trace;

  const statusColor = {
    running: 'badge-yellow',
    completed: 'badge-green',
    failed: 'badge-red',
  }[snapshot.status] || 'badge-gray';

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-bg-secondary rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-border flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className={cn('badge', statusColor)}>{snapshot.status}</span>
              <span className="badge badge-purple">{snapshot.subagent_model}</span>
              {snapshot.subagent_name && (
                <span className="text-sm text-text-muted">{snapshot.subagent_name}</span>
              )}
            </div>
            <div className="text-lg font-semibold text-text-primary">
              {snapshot.task || snapshot.research_topic || 'Subagent Execution'}
            </div>
          </div>
          <button
            onClick={onClose}
            className="btn btn-ghost text-xl"
            title="Close"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Summary Stats */}
          {trace && (
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="panel p-3 text-center">
                <div className="text-2xl font-bold text-accent-blue">{trace.llm_call_count}</div>
                <div className="text-xs text-text-muted">LLM Calls</div>
              </div>
              <div className="panel p-3 text-center">
                <div className="text-2xl font-bold text-accent-green">{trace.tool_call_count}</div>
                <div className="text-xs text-text-muted">Tool Calls</div>
              </div>
              <div className="panel p-3 text-center">
                <div className="text-2xl font-bold text-accent-purple">{trace.total_tokens}</div>
                <div className="text-xs text-text-muted">Total Tokens</div>
              </div>
              <div className="panel p-3 text-center">
                <div className="text-2xl font-bold text-accent-yellow">
                  {formatDuration(trace.total_duration_ms)}
                </div>
                <div className="text-xs text-text-muted">Duration</div>
              </div>
            </div>
          )}

          {/* Context Provided */}
          {(trace?.context_injected || trace?.user_message) && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-text-primary mb-2">Context Provided</h3>
              {trace.context_injected && (
                <div className="mb-2">
                  <div className="text-xs text-text-muted mb-1">Injected Context:</div>
                  <pre className="text-xs text-text-secondary bg-bg-tertiary p-3 rounded overflow-x-auto max-h-40 overflow-y-auto whitespace-pre-wrap">
                    {trace.context_injected}
                  </pre>
                </div>
              )}
              {trace.user_message && (
                <div>
                  <div className="text-xs text-text-muted mb-1">User Message:</div>
                  <pre className="text-xs text-text-secondary bg-bg-tertiary p-3 rounded overflow-x-auto max-h-40 overflow-y-auto whitespace-pre-wrap">
                    {trace.user_message}
                  </pre>
                </div>
              )}
            </div>
          )}

          {/* Event Timeline */}
          {trace && (trace.llm_calls.length > 0 || trace.tool_calls.length > 0) && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-text-primary mb-2">Execution Timeline</h3>

              {/* Combine and sort events by timestamp */}
              {(() => {
                const events: Array<{ type: 'llm' | 'tool'; data: SubagentLLMCall | SubagentToolCall; timestamp: string }> = [
                  ...trace.llm_calls.map(c => ({ type: 'llm' as const, data: c, timestamp: c.timestamp })),
                  ...trace.tool_calls.map(c => ({ type: 'tool' as const, data: c, timestamp: c.timestamp })),
                ].sort((a, b) => a.timestamp.localeCompare(b.timestamp));

                let llmIndex = 0;
                let toolIndex = 0;

                return events.map((event, i) => {
                  if (event.type === 'llm') {
                    return <LLMCallItem key={`llm-${i}`} call={event.data as SubagentLLMCall} index={llmIndex++} />;
                  } else {
                    return <ToolCallItem key={`tool-${i}`} call={event.data as SubagentToolCall} index={toolIndex++} />;
                  }
                });
              })()}
            </div>
          )}

          {/* Final Output */}
          {(trace?.final_output || snapshot.output) && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-text-primary mb-2">Final Output</h3>
              <pre className="text-sm text-text-secondary bg-bg-tertiary p-3 rounded overflow-x-auto max-h-60 overflow-y-auto whitespace-pre-wrap">
                {trace?.final_output || snapshot.output}
              </pre>
            </div>
          )}

          {/* Error Message */}
          {trace?.error_message && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-accent-red mb-2">Error</h3>
              <div className="text-sm text-accent-red bg-accent-red/10 p-3 rounded">
                {trace.error_message}
              </div>
            </div>
          )}

          {/* System Prompt (collapsible) */}
          {trace?.system_prompt && (
            <details className="mb-6">
              <summary className="text-sm font-semibold text-text-primary cursor-pointer hover:text-accent-blue">
                System Prompt
              </summary>
              <pre className="text-xs text-text-secondary bg-bg-tertiary p-3 rounded overflow-x-auto max-h-60 overflow-y-auto whitespace-pre-wrap mt-2">
                {trace.system_prompt}
              </pre>
            </details>
          )}

          {/* No Trace Available */}
          {!trace && (
            <div className="text-center text-text-muted py-8">
              <div className="text-4xl mb-2">📊</div>
              <div>No execution trace available for this subagent.</div>
              <div className="text-xs mt-1">Traces are only captured for subagents run after this update.</div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border flex justify-between items-center text-xs text-text-muted">
          <div>
            Snapshot ID: {snapshot.snapshot_id}
          </div>
          <div>
            {snapshot.timestamp}
          </div>
        </div>
      </div>
    </div>
  );
}
