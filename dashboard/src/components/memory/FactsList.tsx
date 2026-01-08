import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { formatRelativeTime } from '../../utils';

export function FactsList() {
  const { memoryFacts } = useDashboardStore();
  const { deleteFact, requestMemoryFacts } = useWebSocket();

  const handleDelete = (key: string) => {
    if (confirm(`Delete fact "${key}"?`)) {
      deleteFact(key);
      setTimeout(requestMemoryFacts, 500);
    }
  };

  return (
    <div className="p-4">
      {memoryFacts.length === 0 ? (
        <div className="text-center text-text-muted py-8">
          No facts stored yet. Facts are learned from conversations.
        </div>
      ) : (
        <div className="space-y-2">
          {memoryFacts.map((fact) => (
            <div
              key={fact.key}
              className="panel p-3 flex items-start justify-between gap-3"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-medium text-accent-blue">
                    {fact.key}
                  </span>
                  <span className="text-xs text-text-muted">
                    {formatRelativeTime(fact.created_at)}
                  </span>
                </div>
                <div className="text-sm text-text-secondary">
                  {typeof fact.value === 'object' ? JSON.stringify(fact.value) : String(fact.value ?? '')}
                </div>
              </div>
              <button
                onClick={() => handleDelete(fact.key)}
                className="btn btn-ghost btn-sm text-accent-red shrink-0"
                title="Delete fact"
              >
                🗑️
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
