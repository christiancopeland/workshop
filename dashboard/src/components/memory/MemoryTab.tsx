import { useEffect, useState } from 'react';
import { useDashboardStore } from '../../store';
import { useWebSocket } from '../../hooks/useWebSocket';
import { SemanticSearch } from './SemanticSearch';
import { KnowledgeTimeline } from './KnowledgeTimeline';
import { ProjectNotes } from './ProjectNotes';
import { ProjectsList } from './ProjectsList';
import { UserProfile } from './UserProfile';

type MemorySection = 'timeline' | 'profile' | 'projects' | 'notes';

export function MemoryTab() {
  const { memoryStats } = useDashboardStore();
  const {
    requestMemoryStats,
    requestMemoryFacts,
    requestMemoryProjects,
    requestMemoryProfile,
    requestProjectNotes,
    requestActiveProject,
  } = useWebSocket();

  const [activeSection, setActiveSection] = useState<MemorySection>('timeline');

  // Fetch stats and active project on mount
  useEffect(() => {
    requestMemoryStats();
    requestActiveProject();
  }, [requestMemoryStats, requestActiveProject]);

  const sections: { id: MemorySection; label: string; icon: string }[] = [
    { id: 'timeline', label: 'Timeline', icon: '📚' },
    { id: 'profile', label: 'Profile', icon: '👤' },
    { id: 'projects', label: 'Projects', icon: '📁' },
    { id: 'notes', label: 'Notes', icon: '📝' },
  ];

  const handleSectionChange = (section: MemorySection) => {
    setActiveSection(section);
    switch (section) {
      case 'timeline':
        requestMemoryFacts();
        requestProjectNotes();
        break;
      case 'projects':
        requestMemoryProjects();
        break;
      case 'profile':
        requestMemoryProfile();
        break;
      case 'notes':
        requestActiveProject();
        requestProjectNotes();
        break;
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header with Search */}
      <div className="p-4 border-b border-border bg-bg-secondary">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-text-primary">🧠 Knowledge Base</h2>
          <button onClick={requestMemoryStats} className="btn btn-ghost btn-sm">
            🔄 Refresh
          </button>
        </div>

        {/* Prominent Search */}
        <SemanticSearch />

        {/* Stats Row */}
        {memoryStats && (
          <div className="flex gap-4 mt-3 text-xs text-text-muted">
            <span>{memoryStats.semantic_count} memories</span>
            <span>{memoryStats.facts_count} facts</span>
            <span>{memoryStats.projects_count} projects</span>
          </div>
        )}
      </div>

      {/* Section Tabs */}
      <div className="flex border-b border-border bg-bg-secondary">
        {sections.map((section) => (
          <button
            key={section.id}
            onClick={() => handleSectionChange(section.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeSection === section.id
                ? 'text-accent-blue border-accent-blue'
                : 'text-text-secondary border-transparent hover:text-text-primary'
            }`}
          >
            <span className="mr-1">{section.icon}</span>
            {section.label}
          </button>
        ))}
      </div>

      {/* Section Content */}
      <div className="flex-1 overflow-y-auto">
        {activeSection === 'timeline' && <KnowledgeTimeline />}
        {activeSection === 'profile' && <UserProfile />}
        {activeSection === 'projects' && <ProjectsList />}
        {activeSection === 'notes' && <ProjectNotes />}
      </div>
    </div>
  );
}
