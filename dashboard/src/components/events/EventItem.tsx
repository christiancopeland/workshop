import { DashboardEvent, EVENT_STYLES } from '../../types';
import { formatTimestamp, getEventSummary, cn } from '../../utils';

export interface EventItemProps {
  event: DashboardEvent;
  onClick: () => void;
  compact?: boolean;
}

interface LLMMessage {
  role: string;
  content: string;
}

function LLMContextPreview({ data }: { data: Record<string, unknown> }) {
  const messageCount = typeof data.message_count === 'number' ? data.message_count : undefined;
  const promptLength = typeof data.system_prompt_length === 'number' ? data.system_prompt_length : undefined;
  const messages = Array.isArray(data.messages) ? data.messages as LLMMessage[] : undefined;
  const hasContext = messages && messages.length > 0;

  // Find first user message for preview
  const userMsg = messages?.find(m => m.role === 'user');
  const preview = userMsg?.content
    ? (userMsg.content.length > 80 ? userMsg.content.substring(0, 80) + '...' : userMsg.content)
    : undefined;

  return (
    <div className="mt-2 pt-2 border-t border-border-primary/30">
      <div className="flex flex-wrap gap-3 text-xs text-text-muted">
        {messageCount !== undefined && (
          <span className="flex items-center gap-1">
            <span className="text-accent-blue">💬</span>
            <span>{messageCount} msgs</span>
          </span>
        )}
        {promptLength !== undefined && promptLength > 0 && (
          <span className="flex items-center gap-1">
            <span className="text-accent-purple">��</span>
            <span>{promptLength.toLocaleString()} char prompt</span>
          </span>
        )}
        {hasContext && (
          <span className="flex items-center gap-1 text-accent-green">
            <span>✓</span>
            <span>context logged</span>
          </span>
        )}
      </div>
      {preview && (
        <div className="mt-1 text-xs text-text-muted italic truncate">
          "{preview}"
        </div>
      )}
    </div>
  );
}

export function EventItem({ event, onClick, compact = false }: EventItemProps) {
  const style = EVENT_STYLES[event.type] || {
    icon: '•',
    color: 'blue',
    bgClass: 'bg-accent-blue/10',
    borderClass: 'border-accent-blue/30',
  };

  const summary = getEventSummary(event.type, event.data);
  const toolName = event.data.tool_name as string | undefined;
  const modelName = event.data.model as string | undefined;

  return (
    <div
      className={cn(
        'event-item animate-fade-in',
        style.bgClass,
        'border-l-2',
        style.borderClass,
        compact && 'p-2'
      )}
      onClick={onClick}
    >
      <div className="flex items-start gap-2">
        {/* Icon */}
        <span className={cn('shrink-0', compact ? 'text-sm' : 'text-lg')}>{style.icon}</span>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span className={cn('font-medium text-text-primary capitalize', compact ? 'text-xs' : 'text-sm')}>
              {event.type.replace(/_/g, ' ')}
            </span>
            <span className="text-xs text-text-muted shrink-0 font-mono">
              {formatTimestamp(event.timestamp)}
            </span>
          </div>
          {summary && !compact && (
            <div className="text-sm text-text-secondary mt-1 truncate">
              {summary}
            </div>
          )}
        </div>
      </div>

      {/* Tool/Model info - hide in compact mode */}
      {!compact && (toolName || modelName) && (
        <div className="mt-2 flex flex-wrap gap-2">
          {toolName && (
            <span className="badge badge-yellow">{toolName}</span>
          )}
          {modelName && (
            <span className="badge badge-purple">{modelName}</span>
          )}
        </div>
      )}

      {/* LLM Context Preview - hide in compact mode */}
      {!compact && event.type === 'llm_calling' && (
        <LLMContextPreview data={event.data} />
      )}
    </div>
  );
}
