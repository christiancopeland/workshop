import { useDashboardStore } from '../../store';
import { cn } from '../../utils';
import type { EventFilter } from '../../types';

const FILTERS: { id: EventFilter; label: string; icon: string }[] = [
  { id: 'all', label: 'All', icon: '📡' },
  { id: 'tools', label: 'Tools', icon: '🔧' },
  { id: 'llm', label: 'LLM', icon: '🧠' },
  { id: 'errors', label: 'Errors', icon: '❌' },
];

export function EventFilters() {
  const { eventFilter, setEventFilter, eventSearch, setEventSearch, events, clearEvents } =
    useDashboardStore();

  return (
    <div className="border-b border-border bg-bg-secondary p-3">
      <div className="flex items-center gap-3">
        {/* Filter buttons */}
        <div className="flex items-center gap-1">
          {FILTERS.map((filter) => (
            <button
              key={filter.id}
              onClick={() => setEventFilter(filter.id)}
              className={cn(
                'px-3 py-1.5 text-xs font-medium rounded transition-colors',
                eventFilter === filter.id
                  ? 'bg-accent-blue text-white'
                  : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
              )}
            >
              <span className="mr-1">{filter.icon}</span>
              {filter.label}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="flex-1">
          <input
            type="text"
            value={eventSearch}
            onChange={(e) => setEventSearch(e.target.value)}
            placeholder="Search events..."
            className="input py-1.5 text-sm"
          />
        </div>

        {/* Event count and clear */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-muted">{events.length} events</span>
          <button
            onClick={clearEvents}
            className="btn btn-ghost btn-sm"
            title="Clear all events"
          >
            🗑️
          </button>
        </div>
      </div>
    </div>
  );
}
