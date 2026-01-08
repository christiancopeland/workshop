import { useEffect } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { ResearchItem } from './ResearchItem';
import { ResearchDetail } from './ResearchDetail';

export function ResearchTab() {
  const { researchLibrary, selectedResearch } = useDashboardStore();
  const { requestResearchLibrary } = useWebSocket();

  useEffect(() => {
    requestResearchLibrary();
  }, [requestResearchLibrary]);

  const activeCount = researchLibrary.filter((r) => r.is_active).length;

  return (
    <div className="h-full flex">
      {/* Library List */}
      <div className="w-1/3 border-r border-border overflow-y-auto">
        <div className="p-4 border-b border-border bg-bg-secondary">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-text-primary">
              🔬 Research Library
            </h2>
            <button onClick={requestResearchLibrary} className="btn btn-ghost btn-sm">
              🔄
            </button>
          </div>
          <div className="text-xs text-text-muted mt-1">
            {researchLibrary.length} platforms • {activeCount} active
          </div>
        </div>
        <div>
          {researchLibrary.length === 0 ? (
            <div className="p-4 text-text-muted text-sm">
              No research platforms yet. Research will appear here when the agent
              gathers information.
            </div>
          ) : (
            researchLibrary.map((platform) => (
              <ResearchItem key={platform.filename} platform={platform} />
            ))
          )}
        </div>
      </div>

      {/* Detail View */}
      <div className="flex-1 overflow-y-auto">
        {selectedResearch ? (
          <ResearchDetail platform={selectedResearch} />
        ) : (
          <div className="flex items-center justify-center h-full text-text-muted">
            Select a research platform to view details
          </div>
        )}
      </div>
    </div>
  );
}
