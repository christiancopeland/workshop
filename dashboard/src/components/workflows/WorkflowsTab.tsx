import { useState } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { cn } from '../../utils';

interface SlashCommand {
  name: string;
  description: string;
  icon: string;
  category: 'research' | 'code' | 'memory' | 'system';
}

interface AutomatedWorkflow {
  id: string;
  name: string;
  description: string;
  icon: string;
  steps: string[];
  isRunning?: boolean;
}

const SLASH_COMMANDS: SlashCommand[] = [
  { name: '/research', description: 'Deep research on any topic', icon: '🔬', category: 'research' },
  { name: '/crawl', description: 'Crawl and extract web content', icon: '🕷️', category: 'research' },
  { name: '/summarize', description: 'Summarize research or documents', icon: '📝', category: 'research' },
  { name: '/code', description: 'Generate or analyze code', icon: '💻', category: 'code' },
  { name: '/refactor', description: 'Refactor existing code', icon: '🔧', category: 'code' },
  { name: '/test', description: 'Generate tests for code', icon: '🧪', category: 'code' },
  { name: '/remember', description: 'Store important information', icon: '💾', category: 'memory' },
  { name: '/recall', description: 'Search memory for information', icon: '🔍', category: 'memory' },
  { name: '/forget', description: 'Remove stored information', icon: '🗑️', category: 'memory' },
  { name: '/status', description: 'Show system status', icon: '📊', category: 'system' },
  { name: '/clear', description: 'Clear conversation history', icon: '🧹', category: 'system' },
  { name: '/help', description: 'Show available commands', icon: '❓', category: 'system' },
];

const AUTOMATED_WORKFLOWS: AutomatedWorkflow[] = [
  {
    id: 'deep-research',
    name: 'Deep Research',
    description: 'Multi-source research with synthesis',
    icon: '🔬',
    steps: ['Search multiple sources', 'Extract key information', 'Synthesize findings', 'Generate report'],
  },
  {
    id: 'code-review',
    name: 'Code Review',
    description: 'Comprehensive code analysis',
    icon: '🔍',
    steps: ['Analyze code structure', 'Check for issues', 'Suggest improvements', 'Generate report'],
  },
  {
    id: 'documentation',
    name: 'Auto-Document',
    description: 'Generate documentation for code',
    icon: '📚',
    steps: ['Analyze codebase', 'Extract functions/classes', 'Generate docstrings', 'Create README'],
  },
  {
    id: 'daily-digest',
    name: 'Daily Digest',
    description: 'Summarize recent research and tasks',
    icon: '📰',
    steps: ['Gather recent activities', 'Compile research findings', 'Generate summary', 'Save to memory'],
  },
];

export function WorkflowsTab() {
  const { sendMessage } = useWebSocket();
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [runningWorkflows, setRunningWorkflows] = useState<Set<string>>(new Set());

  const filteredCommands = selectedCategory
    ? SLASH_COMMANDS.filter((cmd) => cmd.category === selectedCategory)
    : SLASH_COMMANDS;

  const handleCommandClick = (command: SlashCommand) => {
    sendMessage(command.name);
  };

  const handleWorkflowStart = (workflow: AutomatedWorkflow) => {
    setRunningWorkflows((prev) => new Set(prev).add(workflow.id));
    sendMessage(`/workflow ${workflow.id}`);
    // Simulate workflow completion after some time (in real implementation, this would be handled by events)
    setTimeout(() => {
      setRunningWorkflows((prev) => {
        const next = new Set(prev);
        next.delete(workflow.id);
        return next;
      });
    }, 5000);
  };

  return (
    <div className="h-full overflow-y-auto p-4 space-y-6">
      {/* Quick Actions */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-text-primary">Slash Commands</h2>
          <div className="flex gap-1">
            <button
              onClick={() => setSelectedCategory(null)}
              className={cn(
                'px-2 py-1 text-xs rounded transition-colors',
                !selectedCategory ? 'bg-accent-blue text-white' : 'text-text-secondary hover:bg-bg-tertiary'
              )}
            >
              All
            </button>
            {['research', 'code', 'memory', 'system'].map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={cn(
                  'px-2 py-1 text-xs rounded capitalize transition-colors',
                  selectedCategory === cat ? 'bg-accent-blue text-white' : 'text-text-secondary hover:bg-bg-tertiary'
                )}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
          {filteredCommands.map((cmd) => (
            <button
              key={cmd.name}
              onClick={() => handleCommandClick(cmd)}
              className="flex items-start gap-3 p-3 bg-bg-secondary border border-border rounded-lg hover:border-accent-blue/50 hover:bg-bg-tertiary/50 transition-all text-left group"
            >
              <span className="text-lg">{cmd.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="font-medium text-text-primary text-sm group-hover:text-accent-blue transition-colors">
                  {cmd.name}
                </div>
                <div className="text-xs text-text-muted truncate">{cmd.description}</div>
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* Automated Workflows */}
      <section>
        <h2 className="text-sm font-semibold text-text-primary mb-3">Automated Workflows</h2>
        <div className="space-y-3">
          {AUTOMATED_WORKFLOWS.map((workflow) => {
            const isRunning = runningWorkflows.has(workflow.id);
            return (
              <div
                key={workflow.id}
                className="bg-bg-secondary border border-border rounded-lg p-4"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{workflow.icon}</span>
                    <div>
                      <h3 className="font-medium text-text-primary">{workflow.name}</h3>
                      <p className="text-xs text-text-muted">{workflow.description}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleWorkflowStart(workflow)}
                    disabled={isRunning}
                    className={cn(
                      'btn btn-sm',
                      isRunning ? 'bg-accent-yellow/20 text-accent-yellow' : 'btn-primary'
                    )}
                  >
                    {isRunning ? (
                      <span className="flex items-center gap-2">
                        <span className="w-3 h-3 border-2 border-accent-yellow border-t-transparent rounded-full animate-spin" />
                        Running
                      </span>
                    ) : (
                      'Start'
                    )}
                  </button>
                </div>
                <div className="flex items-center gap-2 text-xs text-text-muted">
                  {workflow.steps.map((step, i) => (
                    <span key={i} className="flex items-center gap-1">
                      {i > 0 && <span className="text-border">→</span>}
                      <span className={cn(
                        isRunning && i === 0 && 'text-accent-yellow'
                      )}>
                        {step}
                      </span>
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Custom Workflow Builder Placeholder */}
      <section className="border border-dashed border-border rounded-lg p-6 text-center">
        <div className="text-2xl mb-2">🛠️</div>
        <h3 className="font-medium text-text-secondary mb-1">Custom Workflows</h3>
        <p className="text-xs text-text-muted">
          Create custom automated workflows by chaining commands together.
        </p>
        <button className="btn btn-ghost btn-sm mt-3" disabled>
          Coming Soon
        </button>
      </section>
    </div>
  );
}
