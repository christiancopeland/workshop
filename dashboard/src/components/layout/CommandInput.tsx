import { useState, useCallback, KeyboardEvent } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { cn } from '../../utils';

export function CommandInput() {
  const [message, setMessage] = useState('');
  const { agentState } = useDashboardStore();
  const { sendMessage, emergencyStop, clearHistory } = useWebSocket();

  const isProcessing = agentState !== 'idle' && agentState !== 'error';

  const handleSend = useCallback(() => {
    if (!message.trim() || isProcessing) return;
    sendMessage(message.trim());
    setMessage('');
  }, [message, isProcessing, sendMessage]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className="border-t border-border bg-bg-secondary p-4">
      <div className="flex items-center gap-2">
        <div className="flex-1 relative">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isProcessing ? 'Agent is processing...' : 'Send a message...'}
            disabled={isProcessing}
            className="input pr-20"
          />
          {isProcessing && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <div className="spinner" />
            </div>
          )}
        </div>
        <button
          onClick={handleSend}
          disabled={!message.trim() || isProcessing}
          className="btn btn-primary"
        >
          Send
        </button>
        <div className="flex items-center gap-1">
          <button
            onClick={emergencyStop}
            disabled={!isProcessing}
            className={cn(
              'btn btn-sm',
              isProcessing ? 'btn-danger' : 'btn-ghost opacity-50'
            )}
            title="Cancel current operation"
          >
            ⏹
          </button>
          <button
            onClick={clearHistory}
            className="btn btn-ghost btn-sm"
            title="Clear conversation history"
          >
            🗑️
          </button>
        </div>
      </div>
    </div>
  );
}
