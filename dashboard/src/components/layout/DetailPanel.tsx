import { useMemo } from 'react';
import { useDashboardStore } from '../../store';
import { EventItem } from '../events/EventItem';
import { EventFilters } from '../events/EventFilters';
import { EventModal } from '../events/EventModal';

export function DetailPanel() {
  const {
    events,
    eventFilter,
    eventSearch,
    selectedEvent,
    setSelectedEvent,
    currentSession,
    staleState
  } = useDashboardStore();

  const filteredEvents = useMemo(() => {
    let filtered = events;

    // Apply filter
    if (eventFilter !== 'all' && eventFilter !== 'grouped') {
      filtered = filtered.filter((event) => {
        switch (eventFilter) {
          case 'tools':
            return (
              event.type === 'tool_calling' ||
              event.type === 'tool_result' ||
              event.type === 'tool_error'
            );
          case 'llm':
            return event.type === 'llm_calling' || event.type === 'llm_complete';
          case 'errors':
            return (
              event.type === 'error' ||
              event.type === 'tool_error' ||
              event.type === 'warning'
            );
          default:
            return true;
        }
      });
    }

    // Apply search
    if (eventSearch.trim()) {
      const search = eventSearch.toLowerCase();
      filtered = filtered.filter((event) => {
        const typeMatch = event.type.toLowerCase().includes(search);
        const dataMatch = JSON.stringify(event.data).toLowerCase().includes(search);
        return typeMatch || dataMatch;
      });
    }

    return filtered;
  }, [events, eventFilter, eventSearch]);

  return (
    <aside className="dashboard-detail flex flex-col h-full overflow-hidden">
      {/* Compact Header with Session & Warnings */}
      <div className="shrink-0 p-3 border-b border-border bg-bg-secondary">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-text-primary">📡 Event Stream</span>
            <span className="text-xs text-text-muted">
              {events.length} events
            </span>
          </div>
          {currentSession?.id && (
            <span className="text-xs text-text-muted font-mono">
              Session: {currentSession.id.slice(0, 6)}
            </span>
          )}
        </div>

        {/* Stale State Warning - Compact */}
        {staleState?.has_stale && (
          <div className="mt-2 flex items-center gap-2 text-xs text-accent-yellow bg-accent-yellow/10 rounded px-2 py-1">
            <span>⚠️</span>
            <span>{staleState.task_count} stale tasks</span>
          </div>
        )}
      </div>

      {/* Event Filters */}
      <div className="shrink-0 border-b border-border">
        <EventFilters />
      </div>

      {/* Events List */}
      <div className="flex-1 overflow-y-auto">
        {filteredEvents.length === 0 ? (
          <div className="flex items-center justify-center h-full text-text-muted text-sm p-4">
            {events.length === 0
              ? 'No events yet'
              : 'No events match filter'}
          </div>
        ) : (
          <div>
            {filteredEvents.map((event) => (
              <EventItem
                key={event.id}
                event={event}
                onClick={() => setSelectedEvent(event)}
                compact
              />
            ))}
          </div>
        )}
      </div>

      {/* Event Modal */}
      {selectedEvent && (
        <EventModal
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}
    </aside>
  );
}
