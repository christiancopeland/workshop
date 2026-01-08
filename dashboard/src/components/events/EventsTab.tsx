import { useMemo } from 'react';
import { useDashboardStore } from '../../store';
import { EventItem } from './EventItem';
import { EventFilters } from './EventFilters';
import { EventModal } from './EventModal';

export function EventsTab() {
  const { events, eventFilter, eventSearch, selectedEvent, setSelectedEvent } =
    useDashboardStore();

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
    <div className="h-full flex flex-col">
      <EventFilters />
      <div className="flex-1 overflow-y-auto">
        {filteredEvents.length === 0 ? (
          <div className="flex items-center justify-center h-full text-text-muted">
            {events.length === 0
              ? 'No events yet. Send a message to get started.'
              : 'No events match your filter.'}
          </div>
        ) : (
          <div>
            {filteredEvents.map((event) => (
              <EventItem
                key={event.id}
                event={event}
                onClick={() => setSelectedEvent(event)}
              />
            ))}
          </div>
        )}
      </div>
      {selectedEvent && (
        <EventModal
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}
    </div>
  );
}
