import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type {
  DashboardEvent,
  EventFilter,
  EventGroup,
  TaskState,
  ResearchPlatform,
  ResearchDetail,
  MemoryStats,
  Fact,
  Project,
  Message,
  MemorySearchResult,
  Session,
  StaleState,
  AgentState,
  LoadedModel,
  SubagentSnapshot,
  SubagentSnapshotDetail,
  ConnectionStatus,
  TabId,
  ProjectNote,
  ActiveProject,
} from '../types';

const MAX_EVENTS = 500;

interface DashboardStats {
  messages: number;
  toolCalls: number;
  llmCalls: number;
  errors: number;
}

interface DashboardStore {
  // Connection
  connectionStatus: ConnectionStatus;
  setConnectionStatus: (status: ConnectionStatus) => void;

  // Stats
  stats: DashboardStats;
  incrementStat: (stat: keyof DashboardStats) => void;
  resetStats: () => void;

  // Events
  events: DashboardEvent[];
  eventFilter: EventFilter;
  eventSearch: string;
  eventGroups: EventGroup[];
  addEvent: (event: DashboardEvent) => void;
  setEventFilter: (filter: EventFilter) => void;
  setEventSearch: (query: string) => void;
  clearEvents: () => void;
  toggleEventGroup: (groupId: string) => void;

  // Trace
  currentTraceId: string | null;
  setCurrentTraceId: (id: string | null) => void;

  // Tasks
  tasks: TaskState;
  setTasks: (tasks: TaskState) => void;

  // Research
  researchLibrary: ResearchPlatform[];
  selectedResearch: ResearchDetail | null;
  setResearchLibrary: (library: ResearchPlatform[]) => void;
  setSelectedResearch: (research: ResearchDetail | null) => void;

  // Subagents
  subagentHistory: SubagentSnapshot[];
  selectedSubagent: SubagentSnapshotDetail | null;
  setSubagentHistory: (history: SubagentSnapshot[]) => void;
  setSelectedSubagent: (subagent: SubagentSnapshotDetail | null) => void;

  // Memory
  memoryStats: MemoryStats | null;
  memoryFacts: Fact[];
  memoryProjects: Project[];
  memoryProfile: string;
  memoryMessages: Message[];
  memorySearchResults: MemorySearchResult[];
  memorySearchQuery: string;
  lastUserMessage: string;
  activeProject: ActiveProject | null;
  projectNotes: ProjectNote[];
  setMemoryStats: (stats: MemoryStats) => void;
  setMemoryFacts: (facts: Fact[]) => void;
  setMemoryProjects: (projects: Project[]) => void;
  setMemoryProfile: (profile: string) => void;
  setMemoryMessages: (messages: Message[]) => void;
  setMemorySearchResults: (results: MemorySearchResult[]) => void;
  setMemorySearchQuery: (query: string) => void;
  setLastUserMessage: (message: string) => void;
  setActiveProject: (project: ActiveProject | null) => void;
  setProjectNotes: (notes: ProjectNote[]) => void;

  // Session
  currentSession: Session | null;
  sessionHistory: Session[];
  staleState: StaleState | null;
  setCurrentSession: (session: Session | null) => void;
  setSessionHistory: (sessions: Session[]) => void;
  setStaleState: (state: StaleState | null) => void;

  // Agent
  agentState: AgentState;
  agentDetails: string;
  agentModel: string | null;
  agentTimerStart: number | null;
  setAgentState: (state: AgentState, details?: string) => void;
  setAgentModel: (model: string | null) => void;
  setAgentTimerStart: (start: number | null) => void;

  // System
  vramUsage: { used: number; total: number; percent: number } | null;
  loadedModels: LoadedModel[];
  setSystemStatus: (vram: { used: number; total: number; percent: number } | null, models: LoadedModel[]) => void;

  // UI
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;
  recentSkills: string[];
  addRecentSkill: (skill: string) => void;
  activeTools: string[];
  setActiveTools: (tools: string[]) => void;

  // Modal
  selectedEvent: DashboardEvent | null;
  setSelectedEvent: (event: DashboardEvent | null) => void;

  // Persistence
  persistedSessionId: string | null;
  setPersistedSessionId: (id: string | null) => void;
  clearPersistedState: () => void;
}

export const useDashboardStore = create<DashboardStore>()(
  persist(
    immer((set) => ({
    // Connection
    connectionStatus: 'disconnected',
    setConnectionStatus: (status) =>
      set((state) => {
        state.connectionStatus = status;
      }),

    // Stats
    stats: { messages: 0, toolCalls: 0, llmCalls: 0, errors: 0 },
    incrementStat: (stat) =>
      set((state) => {
        state.stats[stat]++;
      }),
    resetStats: () =>
      set((state) => {
        state.stats = { messages: 0, toolCalls: 0, llmCalls: 0, errors: 0 };
      }),

    // Events
    events: [],
    eventFilter: 'all',
    eventSearch: '',
    eventGroups: [],
    addEvent: (event) =>
      set((state) => {
        state.events.unshift(event);
        if (state.events.length > MAX_EVENTS) {
          state.events.pop();
        }
        // Update stats based on event type
        if (event.type === 'user_input' || event.type === 'assistant_response') {
          state.stats.messages++;
        } else if (event.type === 'tool_calling' || event.type === 'tool_result') {
          state.stats.toolCalls++;
        } else if (event.type === 'llm_calling' || event.type === 'llm_complete') {
          state.stats.llmCalls++;
        } else if (event.type === 'error' || event.type === 'tool_error') {
          state.stats.errors++;
        }
        // Update trace
        if (event.trace_id) {
          state.currentTraceId = event.trace_id;
        }
      }),
    setEventFilter: (filter) =>
      set((state) => {
        state.eventFilter = filter;
      }),
    setEventSearch: (query) =>
      set((state) => {
        state.eventSearch = query;
      }),
    clearEvents: () =>
      set((state) => {
        state.events = [];
        state.eventGroups = [];
      }),
    toggleEventGroup: (groupId) =>
      set((state) => {
        const group = state.eventGroups.find((g) => g.id === groupId);
        if (group) {
          group.collapsed = !group.collapsed;
        }
      }),

    // Trace
    currentTraceId: null,
    setCurrentTraceId: (id) =>
      set((state) => {
        state.currentTraceId = id;
      }),

    // Tasks
    tasks: {
      has_tasks: false,
      tasks: [],
      original_request: '',
      stats: { total: 0, completed: 0, in_progress: 0, pending: 0 },
    },
    setTasks: (tasks) =>
      set((state) => {
        state.tasks = tasks;
      }),

    // Research
    researchLibrary: [],
    selectedResearch: null,
    setResearchLibrary: (library) =>
      set((state) => {
        state.researchLibrary = library;
      }),
    setSelectedResearch: (research) =>
      set((state) => {
        state.selectedResearch = research;
      }),

    // Subagents
    subagentHistory: [],
    selectedSubagent: null,
    setSubagentHistory: (history) =>
      set((state) => {
        state.subagentHistory = history;
      }),
    setSelectedSubagent: (subagent) =>
      set((state) => {
        state.selectedSubagent = subagent;
      }),

    // Memory
    memoryStats: null,
    memoryFacts: [],
    memoryProjects: [],
    memoryProfile: '',
    memoryMessages: [],
    memorySearchResults: [],
    memorySearchQuery: '',
    lastUserMessage: '',
    activeProject: null,
    projectNotes: [],
    setMemoryStats: (stats) =>
      set((state) => {
        state.memoryStats = stats;
      }),
    setMemoryFacts: (facts) =>
      set((state) => {
        state.memoryFacts = facts;
      }),
    setMemoryProjects: (projects) =>
      set((state) => {
        state.memoryProjects = projects;
      }),
    setMemoryProfile: (profile) =>
      set((state) => {
        state.memoryProfile = profile;
      }),
    setMemoryMessages: (messages) =>
      set((state) => {
        state.memoryMessages = messages;
      }),
    setMemorySearchResults: (results) =>
      set((state) => {
        state.memorySearchResults = results;
      }),
    setMemorySearchQuery: (query) =>
      set((state) => {
        state.memorySearchQuery = query;
      }),
    setLastUserMessage: (message) =>
      set((state) => {
        state.lastUserMessage = message;
      }),
    setActiveProject: (project) =>
      set((state) => {
        state.activeProject = project;
      }),
    setProjectNotes: (notes) =>
      set((state) => {
        state.projectNotes = notes;
      }),

    // Session
    currentSession: null,
    sessionHistory: [],
    staleState: null,
    setCurrentSession: (session) =>
      set((state) => {
        state.currentSession = session;
      }),
    setSessionHistory: (sessions) =>
      set((state) => {
        state.sessionHistory = sessions;
      }),
    setStaleState: (stale) =>
      set((state) => {
        state.staleState = stale;
      }),

    // Agent
    agentState: 'idle',
    agentDetails: '',
    agentModel: null,
    agentTimerStart: null,
    setAgentState: (state, details = '') =>
      set((s) => {
        s.agentState = state;
        s.agentDetails = details;
        if (state !== 'idle') {
          s.agentTimerStart = s.agentTimerStart || Date.now();
        } else {
          s.agentTimerStart = null;
        }
      }),
    setAgentModel: (model) =>
      set((state) => {
        state.agentModel = model;
      }),
    setAgentTimerStart: (start) =>
      set((state) => {
        state.agentTimerStart = start;
      }),

    // System
    vramUsage: null,
    loadedModels: [],
    setSystemStatus: (vram, models) =>
      set((state) => {
        state.vramUsage = vram;
        state.loadedModels = models;
      }),

    // UI
    activeTab: 'workflows',
    setActiveTab: (tab) =>
      set((state) => {
        state.activeTab = tab;
      }),
    recentSkills: [],
    addRecentSkill: (skill) =>
      set((state) => {
        if (!state.recentSkills.includes(skill)) {
          state.recentSkills.unshift(skill);
          if (state.recentSkills.length > 10) {
            state.recentSkills.pop();
          }
        }
      }),
    activeTools: [],
    setActiveTools: (tools) =>
      set((state) => {
        state.activeTools = tools;
      }),

    // Modal
    selectedEvent: null,
    setSelectedEvent: (event) =>
      set((state) => {
        state.selectedEvent = event;
      }),

    // Persistence
    persistedSessionId: null,
    setPersistedSessionId: (id) =>
      set((state) => {
        state.persistedSessionId = id;
      }),
    clearPersistedState: () =>
      set((state) => {
        state.events = [];
        state.stats = { messages: 0, toolCalls: 0, llmCalls: 0, errors: 0 };
        state.tasks = {
          has_tasks: false,
          tasks: [],
          original_request: '',
          stats: { total: 0, completed: 0, in_progress: 0, pending: 0 },
        };
        state.currentTraceId = null;
        state.subagentHistory = [];
        state.researchLibrary = [];
        state.memoryFacts = [];
        state.memoryProjects = [];
        state.memoryProfile = '';
        state.activeProject = null;
        state.projectNotes = [];
        state.recentSkills = [];
        state.persistedSessionId = null;
      }),
  })),
    {
      name: 'workshop-dashboard',
      version: 1, // Increment when changing persisted state shape
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        // Only persist these fields - survives refresh/lock
        events: state.events,
        stats: state.stats,
        tasks: state.tasks,
        currentTraceId: state.currentTraceId,
        currentSession: state.currentSession,
        subagentHistory: state.subagentHistory,
        researchLibrary: state.researchLibrary,
        memoryFacts: state.memoryFacts,
        memoryProjects: state.memoryProjects,
        memoryProfile: state.memoryProfile,
        recentSkills: state.recentSkills,
        persistedSessionId: state.persistedSessionId,
      }),
      // Migrate from old versions or fix malformed data
      migrate: (persistedState: unknown, version: number) => {
        const state = persistedState as Partial<DashboardStore>;

        // Version 0 -> 1: Fix session_id -> id mapping
        if (version === 0) {
          // Clear session with wrong field name
          if (state.currentSession && !state.currentSession.id) {
            const rawSession = state.currentSession as unknown as { session_id?: string };
            if (rawSession.session_id) {
              state.currentSession = {
                ...state.currentSession,
                id: rawSession.session_id,
              };
            } else {
              state.currentSession = null;
            }
          }
        }

        return state as DashboardStore;
      },
    }
  )
);
