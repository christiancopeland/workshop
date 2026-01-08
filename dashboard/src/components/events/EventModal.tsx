import { DashboardEvent, EVENT_STYLES } from '../../types';
import { formatTimestamp, formatJSON } from '../../utils';
import { useCallback, useState } from 'react';

interface EventModalProps {
  event: DashboardEvent;
  onClose: () => void;
}

interface LLMMessage {
  role: string;
  content: string;
  full_length?: number;
}

function LLMMessageView({ message, index }: { message: LLMMessage; index: number }) {
  const [expanded, setExpanded] = useState(false);

  const roleColors: Record<string, string> = {
    system: 'bg-accent-purple/20 border-accent-purple/40 text-accent-purple',
    user: 'bg-accent-blue/20 border-accent-blue/40 text-accent-blue',
    assistant: 'bg-accent-green/20 border-accent-green/40 text-accent-green',
    tool: 'bg-accent-yellow/20 border-accent-yellow/40 text-accent-yellow',
  };

  const roleIcons: Record<string, string> = {
    system: '⚙️',
    user: '👤',
    assistant: '🤖',
    tool: '🔧',
  };

  const colorClass = roleColors[message.role] || 'bg-bg-tertiary border-border-primary text-text-secondary';
  const icon = roleIcons[message.role] || '📝';
  const isTruncated = message.full_length && message.full_length > message.content.length;

  return (
    <div className={`rounded-lg border p-3 mb-2 ${colorClass}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span>{icon}</span>
          <span className="font-semibold capitalize text-sm">{message.role}</span>
          <span className="text-xs opacity-60">#{index + 1}</span>
        </div>
        <div className="flex items-center gap-2">
          {message.full_length && (
            <span className="text-xs opacity-60">{message.full_length.toLocaleString()} chars</span>
          )}
          {isTruncated && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-xs underline opacity-70 hover:opacity-100"
            >
              {expanded ? 'Collapse' : 'Expand'}
            </button>
          )}
        </div>
      </div>
      <div className={`text-sm whitespace-pre-wrap break-words ${expanded ? '' : 'max-h-48 overflow-hidden'}`}>
        {message.content}
      </div>
      {!expanded && isTruncated && (
        <div className="text-xs opacity-50 mt-1 italic">
          Content truncated. Click "Expand" to see more.
        </div>
      )}
    </div>
  );
}

function LLMContextView({ data }: { data: Record<string, unknown> }) {
  const messages = data.messages as LLMMessage[] | undefined;
  const systemPrompt = data.system_prompt as string | undefined;
  const systemPromptLength = data.system_prompt_length as number | undefined;
  const model = data.model as string | undefined;
  const messageCount = data.message_count as number | undefined;

  return (
    <div className="space-y-4">
      {/* Summary Header */}
      <div className="bg-bg-tertiary rounded-lg p-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-text-muted mb-1">Model</div>
            <div className="text-sm font-mono font-semibold text-accent-purple">{model || 'Unknown'}</div>
          </div>
          <div>
            <div className="text-xs text-text-muted mb-1">Message Count</div>
            <div className="text-sm font-mono">{messageCount || 0}</div>
          </div>
          {systemPromptLength && systemPromptLength > 0 && (
            <div className="col-span-2">
              <div className="text-xs text-text-muted mb-1">System Prompt Length</div>
              <div className="text-sm font-mono">{systemPromptLength.toLocaleString()} chars</div>
            </div>
          )}
        </div>
      </div>

      {/* System Prompt (if separate from messages) */}
      {systemPrompt && (
        <div>
          <div className="text-xs text-text-muted mb-2 flex items-center gap-2">
            <span>⚙️</span>
            <span>System Prompt</span>
          </div>
          <div className="bg-accent-purple/10 border border-accent-purple/30 rounded-lg p-3">
            <pre className="text-sm whitespace-pre-wrap break-words text-text-secondary max-h-48 overflow-y-auto">
              {systemPrompt}
            </pre>
          </div>
        </div>
      )}

      {/* Messages */}
      {messages && messages.length > 0 ? (
        <div>
          <div className="text-xs text-text-muted mb-2 flex items-center gap-2">
            <span>💬</span>
            <span>Messages ({messages.length})</span>
          </div>
          <div className="space-y-2 max-h-[40vh] overflow-y-auto pr-2">
            {messages.map((msg, idx) => (
              <LLMMessageView key={idx} message={msg} index={idx} />
            ))}
          </div>
        </div>
      ) : (
        <div className="text-sm text-text-muted italic">
          No message context available for this LLM call.
        </div>
      )}
    </div>
  );
}

export function EventModal({ event, onClose }: EventModalProps) {
  const style = EVENT_STYLES[event.type] || {
    icon: '•',
    color: 'blue',
    bgClass: 'bg-accent-blue/10',
    borderClass: 'border-accent-blue/30',
  };

  const [showRaw, setShowRaw] = useState(false);
  const isLLMEvent = event.type === 'llm_calling';

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(formatJSON(event));
  }, [event]);

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <div className="modal-overlay" onClick={handleBackdropClick}>
      <div className={`modal-content ${isLLMEvent ? 'max-w-4xl' : ''}`}>
        {/* Header */}
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <span className="text-xl">{style.icon}</span>
            <span className="panel-title capitalize">
              {event.type.replace(/_/g, ' ')}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {isLLMEvent && (
              <button
                onClick={() => setShowRaw(!showRaw)}
                className={`btn btn-ghost btn-sm ${showRaw ? 'bg-bg-tertiary' : ''}`}
              >
                {showRaw ? '📊 Formatted' : '{ } Raw'}
              </button>
            )}
            <button onClick={handleCopy} className="btn btn-ghost btn-sm">
              📋 Copy
            </button>
            <button onClick={onClose} className="btn btn-ghost btn-sm">
              ✕
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="panel-content overflow-y-auto max-h-[70vh]">
          {/* Metadata */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <div className="text-xs text-text-muted mb-1">Timestamp</div>
              <div className="text-sm font-mono">{formatTimestamp(event.timestamp)}</div>
            </div>
            <div>
              <div className="text-xs text-text-muted mb-1">Event ID</div>
              <div className="text-sm font-mono text-text-secondary truncate">
                {event.id}
              </div>
            </div>
            {event.trace_id && (
              <div className="col-span-2">
                <div className="text-xs text-text-muted mb-1">Trace ID</div>
                <div className="text-sm font-mono text-text-secondary">
                  {event.trace_id}
                </div>
              </div>
            )}
          </div>

          {/* Event Data - LLM-specific or Raw */}
          {isLLMEvent && !showRaw ? (
            <LLMContextView data={event.data} />
          ) : (
            <div>
              <div className="text-xs text-text-muted mb-2">Event Data</div>
              <pre className="bg-bg-tertiary rounded p-4 text-sm text-text-secondary overflow-x-auto">
                {formatJSON(event.data)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
