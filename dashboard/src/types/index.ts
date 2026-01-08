export * from './events';
export * from './tasks';
export * from './research';
export * from './memory';
export * from './session';
export * from './agent';
export * from './subagent';
export * from './api';

// UI-specific types
export type TabId = 'workflows' | 'tasks' | 'research' | 'subagents' | 'memory';

export interface Tab {
  id: TabId;
  label: string;
  icon: string;
}

export const TABS: Tab[] = [
  { id: 'workflows', label: 'Workflows', icon: '⚡' },
  { id: 'tasks', label: 'Tasks', icon: '📋' },
  { id: 'research', label: 'Research', icon: '🔬' },
  { id: 'subagents', label: 'Subagents', icon: '🤖' },
  { id: 'memory', label: 'Memory', icon: '🧠' },
];
