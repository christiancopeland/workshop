import { useDashboardStore } from '../../store';
import { formatTimestamp, cn } from '../../utils';

export function ConversationHistory() {
  const { memoryMessages } = useDashboardStore();

  return (
    <div className="p-4">
      {memoryMessages.length === 0 ? (
        <div className="text-center text-text-muted py-8">
          No conversation history available.
        </div>
      ) : (
        <div className="space-y-3">
          {memoryMessages.map((message, i) => (
            <div
              key={i}
              className={cn(
                'panel p-3',
                message.role === 'user' && 'border-l-2 border-l-accent-blue',
                message.role === 'assistant' && 'border-l-2 border-l-accent-green',
                message.role === 'system' && 'border-l-2 border-l-accent-purple'
              )}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-lg">
                    {message.role === 'user' && '👤'}
                    {message.role === 'assistant' && '🤖'}
                    {message.role === 'system' && '⚙️'}
                  </span>
                  <span
                    className={cn(
                      'text-xs font-medium uppercase',
                      message.role === 'user' && 'text-accent-blue',
                      message.role === 'assistant' && 'text-accent-green',
                      message.role === 'system' && 'text-accent-purple'
                    )}
                  >
                    {message.role}
                  </span>
                </div>
                <span className="text-xs text-text-muted font-mono">
                  {formatTimestamp(message.timestamp)}
                </span>
              </div>
              <div className="text-sm text-text-secondary whitespace-pre-wrap">
                {message.content.length > 500
                  ? message.content.slice(0, 500) + '...'
                  : message.content}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
