import { useDashboardStore } from '../../store';
import { TABS } from '../../types';
import { cn } from '../../utils';
import { WorkflowsTab } from '../workflows/WorkflowsTab';
import { TasksTab } from '../tasks/TasksTab';
import { ResearchTab } from '../research/ResearchTab';
import { SubagentsTab } from '../subagents/SubagentsTab';
import { MemoryTab } from '../memory/MemoryTab';
import { CommandInput } from './CommandInput';

export function MainContent() {
  const { activeTab, setActiveTab } = useDashboardStore();

  return (
    <main className="dashboard-main">
      {/* Tab Navigation */}
      <div className="flex border-b border-border bg-bg-secondary">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'tab',
              activeTab === tab.id && 'tab-active'
            )}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'workflows' && <WorkflowsTab />}
        {activeTab === 'tasks' && <TasksTab />}
        {activeTab === 'research' && <ResearchTab />}
        {activeTab === 'subagents' && <SubagentsTab />}
        {activeTab === 'memory' && <MemoryTab />}
      </div>

      {/* Command Input */}
      <CommandInput />
    </main>
  );
}
