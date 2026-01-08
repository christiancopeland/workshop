import { useState, useCallback, useEffect } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { MarkdownRenderer } from '../ui/MarkdownRenderer';

export function SemanticSearch() {
  const [query, setQuery] = useState('');
  const [showResults, setShowResults] = useState(false);
  const { memorySearchResults, memorySearchQuery, lastUserMessage } = useDashboardStore();
  const { searchSemanticMemory } = useWebSocket();

  // Auto-populate from last user message when component mounts or message changes
  useEffect(() => {
    if (lastUserMessage && !query) {
      setQuery(lastUserMessage);
      // Auto-search with the last message
      searchSemanticMemory(lastUserMessage);
    }
  }, [lastUserMessage]);

  // Show results when search completes
  useEffect(() => {
    if (memorySearchResults.length > 0) {
      setShowResults(true);
    }
  }, [memorySearchResults]);

  const handleSearch = useCallback(() => {
    if (!query.trim()) return;
    searchSemanticMemory(query.trim());
    setShowResults(true);
  }, [query, searchSemanticMemory]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
    if (e.key === 'Escape') {
      setShowResults(false);
    }
  };

  const handleClear = () => {
    setQuery('');
    setShowResults(false);
  };

  return (
    <div className="relative">
      {/* Compact Search Input */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search your knowledge base..."
            className="input w-full py-2 pl-9 pr-9 text-sm"
          />
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted text-sm">
            🔍
          </span>
          {query && (
            <button
              onClick={handleClear}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary text-xs"
              title="Clear"
            >
              ✕
            </button>
          )}
        </div>
        <button onClick={handleSearch} className="btn btn-primary btn-sm px-4">
          Search
        </button>
      </div>

      {/* Results Panel (Collapsible) */}
      {showResults && memorySearchQuery && (
        <div className="absolute left-0 right-0 top-full mt-2 z-50 bg-bg-secondary border border-border rounded-lg shadow-xl max-h-[60vh] overflow-y-auto">
          <div className="sticky top-0 bg-bg-secondary border-b border-border p-3 flex items-center justify-between">
            <span className="text-sm text-text-secondary">
              {memorySearchResults.length} results for "{memorySearchQuery}"
            </span>
            <button
              onClick={() => setShowResults(false)}
              className="btn btn-ghost btn-sm text-xs"
            >
              Close
            </button>
          </div>

          {memorySearchResults.length === 0 ? (
            <div className="text-center text-text-muted py-8">
              <div className="text-2xl mb-2">🔍</div>
              <div className="text-sm">No results found. Try different keywords.</div>
            </div>
          ) : (
            <div className="p-3 space-y-3">
              {memorySearchResults.map((result, i) => (
                <div key={i} className="panel p-3 hover:border-accent-blue/50 transition-colors">
                  {/* Category badge if available */}
                  {result.metadata?.category && (
                    <div className="mb-2">
                      <span className="badge badge-purple text-xs">
                        {String(result.metadata.category)}
                      </span>
                      {result.metadata?.source && (
                        <span className="badge badge-blue text-xs ml-2">
                          {String(result.metadata.source).split('/').pop()}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Content rendered as markdown */}
                  <MarkdownRenderer content={result.content} />

                  {/* Timestamp if available */}
                  {result.metadata?.timestamp && (
                    <div className="mt-2 pt-2 border-t border-border text-xs text-text-muted">
                      Saved: {new Date(String(result.metadata.timestamp)).toLocaleDateString()}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
