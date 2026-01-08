import { useState, useEffect, useMemo } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { MarkdownRenderer } from '../ui/MarkdownRenderer';
import { formatRelativeTime } from '../../utils';
import type { KnowledgeItem } from '../../types';

export function KnowledgeTimeline() {
  const { memoryFacts, projectNotes } = useDashboardStore();
  const {
    requestMemoryFacts,
    requestProjectNotes,
    updateFact,
    deleteFact,
  } = useWebSocket();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [searchFilter, setSearchFilter] = useState('');

  // Fetch data on mount
  useEffect(() => {
    requestMemoryFacts();
    requestProjectNotes();
  }, [requestMemoryFacts, requestProjectNotes]);

  // Combine facts and notes into a unified timeline
  const timelineItems = useMemo((): KnowledgeItem[] => {
    const items: KnowledgeItem[] = [];

    // Add facts
    memoryFacts.forEach((fact) => {
      items.push({
        id: `fact-${fact.key}`,
        type: 'fact',
        title: fact.key,
        content: typeof fact.value === 'string' ? fact.value : JSON.stringify(fact.value, null, 2),
        timestamp: fact.updated_at || fact.created_at,
        category: fact.category,
      });
    });

    // Add project notes
    projectNotes.forEach((note) => {
      items.push({
        id: `note-${note.id}`,
        type: 'note',
        content: note.content,
        timestamp: note.created_at,
        category: note.category,
      });
    });

    // Sort by timestamp (newest first)
    items.sort((a, b) => {
      const dateA = new Date(a.timestamp).getTime();
      const dateB = new Date(b.timestamp).getTime();
      return dateB - dateA;
    });

    return items;
  }, [memoryFacts, projectNotes]);

  // Filter items by search
  const filteredItems = useMemo(() => {
    if (!searchFilter.trim()) return timelineItems;
    const query = searchFilter.toLowerCase();
    return timelineItems.filter(
      (item) =>
        item.content.toLowerCase().includes(query) ||
        item.title?.toLowerCase().includes(query) ||
        item.category?.toLowerCase().includes(query)
    );
  }, [timelineItems, searchFilter]);

  const handleEdit = (item: KnowledgeItem) => {
    setEditingId(item.id);
    setEditContent(item.content);
  };

  const handleSaveEdit = (item: KnowledgeItem) => {
    if (!editContent.trim()) return;
    setIsSaving(true);

    if (item.type === 'fact' && item.title) {
      updateFact(item.title, editContent.trim());
    }

    setTimeout(() => {
      requestMemoryFacts();
      requestProjectNotes();
      setEditingId(null);
      setEditContent('');
      setIsSaving(false);
    }, 500);
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditContent('');
  };

  const handleDelete = (item: KnowledgeItem) => {
    if (!confirm(`Delete this ${item.type}?`)) return;

    if (item.type === 'fact' && item.title) {
      deleteFact(item.title);
    }

    setTimeout(() => {
      requestMemoryFacts();
      requestProjectNotes();
    }, 500);
  };

  const getTypeColor = (type: KnowledgeItem['type']) => {
    switch (type) {
      case 'fact':
        return 'border-l-accent-blue';
      case 'note':
        return 'border-l-accent-green';
      case 'memory':
        return 'border-l-accent-purple';
      default:
        return 'border-l-border';
    }
  };

  const getTypeBadge = (type: KnowledgeItem['type']) => {
    switch (type) {
      case 'fact':
        return <span className="badge badge-blue text-xs">Fact</span>;
      case 'note':
        return <span className="badge badge-green text-xs">Note</span>;
      case 'memory':
        return <span className="badge badge-purple text-xs">Memory</span>;
      default:
        return null;
    }
  };

  return (
    <div className="p-4">
      {/* Search/Filter */}
      <div className="mb-4">
        <div className="relative">
          <input
            type="text"
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            placeholder="Filter timeline..."
            className="input w-full pl-9"
          />
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted text-sm">
            🔍
          </span>
        </div>
      </div>

      {/* Stats */}
      <div className="flex gap-4 mb-4 text-xs text-text-muted">
        <span>{memoryFacts.length} facts</span>
        <span>{projectNotes.length} notes</span>
        {searchFilter && (
          <span className="text-accent-blue">
            {filteredItems.length} matching
          </span>
        )}
      </div>

      {/* Timeline */}
      {filteredItems.length === 0 ? (
        <div className="text-center text-text-muted py-12">
          <div className="text-4xl mb-4">📚</div>
          <div className="text-lg mb-2">Knowledge Timeline</div>
          <div className="text-sm">
            {searchFilter
              ? 'No items match your filter.'
              : 'Facts and notes will appear here as the agent learns.'}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredItems.map((item) => (
            <div
              key={item.id}
              className={`panel p-4 border-l-2 ${getTypeColor(item.type)}`}
            >
              {editingId === item.id ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-text-primary">
                      {item.title || 'Edit'}
                    </span>
                    <div className="text-xs text-text-muted">Supports Markdown</div>
                  </div>
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="input w-full h-32 font-mono text-sm resize-y"
                    autoFocus
                  />
                  <div className="flex gap-2 justify-end">
                    <button
                      onClick={handleCancelEdit}
                      className="btn btn-ghost btn-sm"
                      disabled={isSaving}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => handleSaveEdit(item)}
                      className="btn btn-primary btn-sm"
                      disabled={isSaving}
                    >
                      {isSaving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      {getTypeBadge(item.type)}
                      {item.title && (
                        <span className="text-sm font-medium text-accent-blue">
                          {item.title}
                        </span>
                      )}
                      {item.category && item.category !== 'general' && (
                        <span className="badge badge-purple text-xs">
                          {item.category}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-text-muted">
                        {formatRelativeTime(item.timestamp)}
                      </span>
                      <div className="flex gap-1">
                        <button
                          onClick={() => handleEdit(item)}
                          className="btn btn-ghost btn-sm text-xs px-2"
                          title="Edit"
                        >
                          Edit
                        </button>
                        <button
                          onClick={() => handleDelete(item)}
                          className="btn btn-ghost btn-sm text-xs px-2 text-accent-red"
                          title="Delete"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                  <MarkdownRenderer content={item.content} />
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
