import { useEffect, useCallback, useRef } from 'react';
import { useDashboardStore } from '../store';
import type { OutboundAction, InboundMessage } from '../types/api';
import type { DashboardEvent, EventType } from '../types/events';

const WS_URL = 'ws://localhost:8766';
const RECONNECT_DELAY = 3000;

// Singleton WebSocket state - shared across all hook instances
let globalWs: WebSocket | null = null;
let globalReconnectTimeout: number | null = null;
let globalIsConnecting = false;
let globalMountCount = 0; // Track how many components are using this hook

export function useWebSocket() {
  const connectRef = useRef<() => void>();

  const {
    setConnectionStatus,
    addEvent,
    setTasks,
    setResearchLibrary,
    setSelectedResearch,
    setSubagentHistory,
    setSelectedSubagent,
    setMemoryStats,
    setMemoryFacts,
    setMemoryProjects,
    setMemoryProfile,
    setMemoryMessages,
    setMemorySearchResults,
    setMemorySearchQuery,
    setLastUserMessage,
    setActiveProject,
    setProjectNotes,
    setCurrentSession,
    setSessionHistory,
    setStaleState,
    setAgentState,
    setAgentModel,
    setSystemStatus,
    addRecentSkill,
    setActiveTools,
    resetStats,
    clearEvents,
    persistedSessionId,
    setPersistedSessionId,
    clearPersistedState,
  } = useDashboardStore();

  const handleMessage = useCallback((data: InboundMessage) => {
    // Server sends 'action' field, not 'type'
    const messageType = data.action || data.type;
    switch (messageType) {
      case 'initial_state':
        // Reset and load initial state
        resetStats();
        clearEvents();
        if (data.events) {
          (data.events as DashboardEvent[]).forEach((event) => addEvent(event));
        }
        if (data.tasks) {
          setTasks(data.tasks as any);
        }
        if (data.session) {
          setCurrentSession(data.session as any);
        }
        break;

      case 'event':
        // Backend sends event properties spread directly in the message, not nested under 'event'
        const eventData = data as { type: string; timestamp: number; time_str?: string; trace_id?: string; data?: Record<string, unknown> };
        const event: DashboardEvent = {
          id: `${eventData.trace_id || 'evt'}-${eventData.timestamp}`,
          type: eventData.type as EventType,
          timestamp: eventData.time_str || new Date(eventData.timestamp * 1000).toISOString(),
          trace_id: eventData.trace_id,
          data: eventData.data || {},
        };
        addEvent(event);
        // Capture last user message for semantic search auto-populate
        if (event.type === 'user_input') {
          const text = event.data?.text as string;
          if (text) setLastUserMessage(text);
        }
        // Extract skill from skill_matched events
        if (event.type === 'skill_matched') {
          const skillName = event.data?.skill_name as string;
          if (skillName) addRecentSkill(skillName);
        }
        // Track active tools
        if (event.type === 'tool_calling') {
          const toolName = event.data?.tool_name as string;
          if (toolName) {
            const currentTools = useDashboardStore.getState().activeTools;
            if (!currentTools.includes(toolName)) {
              setActiveTools([...currentTools, toolName]);
            }
          }
        }
        if (event.type === 'tool_result' ||
            event.type === 'tool_error') {
          const toolName = event.data?.tool_name as string;
          if (toolName) {
            const currentTools = useDashboardStore.getState().activeTools;
            setActiveTools(currentTools.filter((t) => t !== toolName));
          }
        }
        break;

      case 'task_list':
        setTasks(data as any);
        break;

      case 'research_library':
        setResearchLibrary(data.platforms as any || []);
        break;

      case 'research_detail':
        setSelectedResearch(data.platform as any);
        break;

      case 'subagent_history':
        setSubagentHistory(data.snapshots as any || []);
        break;

      case 'snapshot_detail':
        setSelectedSubagent(data.snapshot as any || null);
        break;

      case 'system_status':
        setSystemStatus(
          data.vram as any,
          data.loaded_models as any || []
        );
        break;

      case 'memory_stats':
        setMemoryStats(data.stats as any);
        break;

      case 'memory_facts':
        setMemoryFacts(data.facts as any || []);
        break;

      case 'memory_projects':
        setMemoryProjects(data.projects as any || []);
        break;

      case 'memory_profile':
        setMemoryProfile(data.profile as string || '');
        break;

      case 'memory_messages':
        setMemoryMessages(data.messages as any || []);
        break;

      case 'memory_search_results':
        setMemorySearchResults(data.results as any || []);
        break;

      case 'active_project':
        setActiveProject(data.project as any || null);
        break;

      case 'project_notes':
        setProjectNotes(data.notes as any || []);
        break;

      case 'profile_updated':
        // Refresh profile after update
        if (data.success) {
          // Profile will be refreshed by caller
        }
        break;

      case 'note_added':
      case 'note_updated':
      case 'note_deleted':
        // Notes updated - caller should refresh
        break;

      case 'fact_updated':
        // Fact updated - caller should refresh
        break;

      case 'project_set':
        // Project changed - refresh active project
        if (data.success) {
          // Will be refreshed by caller
        }
        break;

      case 'session_info':
      case 'session_started':
      case 'session_resumed':
        {
          const rawSession = data.session as {
            session_id?: string;
            started_at?: string;
            ended_at?: string;
            message_count?: number;
            task_count?: number;
            mode?: string;
          } | undefined;
          const newSessionId = rawSession?.session_id;

          // Check if this is a different session than what we have persisted
          if (newSessionId && persistedSessionId && newSessionId !== persistedSessionId) {
            // Different session - clear stale persisted data
            console.log(`Session changed: ${persistedSessionId} -> ${newSessionId}, clearing persisted state`);
            clearPersistedState();
          }

          // Update persisted session ID
          if (newSessionId) {
            setPersistedSessionId(newSessionId);
          }

          // Map server field names to client types (session_id -> id)
          if (rawSession) {
            setCurrentSession({
              id: rawSession.session_id || '',
              started_at: rawSession.started_at || '',
              ended_at: rawSession.ended_at,
              message_count: rawSession.message_count || 0,
              task_count: rawSession.task_count || 0,
              mode: rawSession.mode,
            });
          } else {
            setCurrentSession(null);
          }
        }
        break;

      case 'session_ended':
        setCurrentSession(null);
        break;

      case 'session_list':
        setSessionHistory(data.sessions as any || []);
        break;

      case 'stale_state':
        setStaleState(data.has_stale ? (data as any) : null);
        break;

      case 'agent_processing':
        setAgentState(
          (data.state as any) || 'thinking',
          data.details as string
        );
        if (data.model) {
          setAgentModel(data.model as string);
        }
        break;

      case 'agent_idle':
        setAgentState('idle');
        break;

      case 'emergency_stop_triggered':
        setAgentState('idle');
        addEvent({
          id: `emergency-${Date.now()}`,
          type: 'warning',
          timestamp: new Date().toISOString(),
          data: { message: 'Emergency stop triggered' },
        });
        break;

      case 'error':
        addEvent({
          id: `error-${Date.now()}`,
          type: 'error',
          timestamp: new Date().toISOString(),
          data: { message: data.message, error: data.error },
        });
        break;
    }
  }, [
    addEvent,
    setTasks,
    setResearchLibrary,
    setSelectedResearch,
    setSubagentHistory,
    setSelectedSubagent,
    setMemoryStats,
    setMemoryFacts,
    setMemoryProjects,
    setMemoryProfile,
    setMemoryMessages,
    setMemorySearchResults,
    setLastUserMessage,
    setActiveProject,
    setProjectNotes,
    setCurrentSession,
    setSessionHistory,
    setStaleState,
    setAgentState,
    setAgentModel,
    setSystemStatus,
    addRecentSkill,
    setActiveTools,
    resetStats,
    clearEvents,
    persistedSessionId,
    setPersistedSessionId,
    clearPersistedState,
  ]);

  const scheduleReconnect = useCallback(() => {
    if (globalReconnectTimeout) return;
    if (globalMountCount === 0) return;

    globalReconnectTimeout = window.setTimeout(() => {
      globalReconnectTimeout = null;
      if (globalMountCount > 0 && connectRef.current) {
        connectRef.current();
      }
    }, RECONNECT_DELAY);
  }, []);

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connection attempts
    if (globalIsConnecting) return;
    if (globalWs?.readyState === WebSocket.OPEN) return;
    if (globalWs?.readyState === WebSocket.CONNECTING) return;

    globalIsConnecting = true;
    setConnectionStatus('reconnecting');

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      globalIsConnecting = false;
      setConnectionStatus('connected');
      // Request initial data - session info first for persistence check
      ws.send(JSON.stringify({ action: 'get_session_info' }));
      ws.send(JSON.stringify({ action: 'get_current_tasks' }));
      ws.send(JSON.stringify({ action: 'get_research_library' }));
      ws.send(JSON.stringify({ action: 'get_system_status' }));
      ws.send(JSON.stringify({ action: 'get_memory_stats' }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onclose = () => {
      globalIsConnecting = false;
      globalWs = null;
      setConnectionStatus('disconnected');
      scheduleReconnect();
    };

    ws.onerror = () => {
      globalIsConnecting = false;
      ws.close();
    };

    globalWs = ws;
  }, [handleMessage, setConnectionStatus, scheduleReconnect]);

  // Keep connectRef updated with latest connect function
  connectRef.current = connect;

  const send = useCallback((action: OutboundAction) => {
    if (globalWs?.readyState === WebSocket.OPEN) {
      globalWs.send(JSON.stringify(action));
    }
  }, []);

  const sendMessage = useCallback((message: string) => {
    send({ action: 'send_message', message });
  }, [send]);

  const emergencyStop = useCallback(() => {
    send({ action: 'emergency_stop' });
  }, [send]);

  const requestTasks = useCallback(() => {
    send({ action: 'get_current_tasks' });
  }, [send]);

  const requestResearchLibrary = useCallback(() => {
    send({ action: 'get_research_library' });
  }, [send]);

  const requestResearchDetail = useCallback((filename: string) => {
    send({ action: 'get_research_detail', filename });
  }, [send]);

  const requestSubagentHistory = useCallback(() => {
    send({ action: 'get_subagent_history' });
  }, [send]);

  const requestSnapshotDetail = useCallback((snapshot_id: string) => {
    send({ action: 'get_snapshot_detail', snapshot_id });
  }, [send]);

  const requestSystemStatus = useCallback(() => {
    send({ action: 'get_system_status' });
  }, [send]);

  const requestMemoryStats = useCallback(() => {
    send({ action: 'get_memory_stats' });
  }, [send]);

  const requestMemoryFacts = useCallback(() => {
    send({ action: 'get_memory_facts' });
  }, [send]);

  const requestMemoryProjects = useCallback(() => {
    send({ action: 'get_memory_projects' });
  }, [send]);

  const requestMemoryProfile = useCallback(() => {
    send({ action: 'get_memory_profile' });
  }, [send]);

  const requestMemoryMessages = useCallback((limit?: number) => {
    send({ action: 'get_memory_messages', limit });
  }, [send]);

  const searchSemanticMemory = useCallback((query: string) => {
    setMemorySearchQuery(query);
    send({ action: 'search_memory', query });
  }, [send, setMemorySearchQuery]);

  const deleteFact = useCallback((key: string) => {
    send({ action: 'delete_memory_fact', key });
  }, [send]);

  const startNewSession = useCallback((mode?: string) => {
    send({ action: 'start_new_session', mode });
  }, [send]);

  const endSession = useCallback((archive?: boolean) => {
    send({ action: 'end_session', archive });
  }, [send]);

  const resumeSession = useCallback((sessionId: string) => {
    send({ action: 'resume_session', session_id: sessionId });
  }, [send]);

  const listSessions = useCallback((limit?: number) => {
    send({ action: 'list_sessions', limit });
  }, [send]);

  const clearStaleState = useCallback(() => {
    send({ action: 'clear_stale_state' });
  }, [send]);

  const clearHistory = useCallback(() => {
    send({ action: 'clear_history' });
  }, [send]);

  // === New Memory Write Methods ===

  const updateProfile = useCallback((profile: string) => {
    send({ action: 'update_user_profile', profile });
  }, [send]);

  const addProjectNote = useCallback((content: string, category?: string) => {
    send({ action: 'add_project_note', content, category: category || 'general' });
  }, [send]);

  const requestProjectNotes = useCallback((projectName?: string) => {
    send({ action: 'get_project_notes', project_name: projectName });
  }, [send]);

  const requestActiveProject = useCallback(() => {
    send({ action: 'get_active_project' });
  }, [send]);

  const setActiveProjectRequest = useCallback((name: string, path?: string, description?: string) => {
    send({ action: 'set_active_project', name, path, description });
  }, [send]);

  const updateFact = useCallback((key: string, value: string) => {
    send({ action: 'update_fact', key, value });
  }, [send]);

  const updateProjectNote = useCallback((noteId: number, content: string) => {
    send({ action: 'update_project_note', note_id: noteId, content });
  }, [send]);

  const deleteProjectNote = useCallback((noteId: number) => {
    send({ action: 'delete_project_note', note_id: noteId });
  }, [send]);

  useEffect(() => {
    globalMountCount++;

    // Only connect if this is the first mount and we're not already connected
    if (globalMountCount === 1 && !globalWs && !globalIsConnecting) {
      connect();
    }

    return () => {
      globalMountCount--;

      // Only disconnect when the last component unmounts
      if (globalMountCount === 0) {
        if (globalReconnectTimeout) {
          clearTimeout(globalReconnectTimeout);
          globalReconnectTimeout = null;
        }
        if (globalWs) {
          globalWs.onclose = null; // Prevent reconnect on intentional close
          globalWs.close();
          globalWs = null;
        }
        globalIsConnecting = false;
      }
    };
  }, [connect]);

  return {
    send,
    sendMessage,
    emergencyStop,
    requestTasks,
    requestResearchLibrary,
    requestResearchDetail,
    requestSubagentHistory,
    requestSnapshotDetail,
    requestSystemStatus,
    requestMemoryStats,
    requestMemoryFacts,
    requestMemoryProjects,
    requestMemoryProfile,
    requestMemoryMessages,
    searchSemanticMemory,
    deleteFact,
    startNewSession,
    endSession,
    resumeSession,
    listSessions,
    clearStaleState,
    clearHistory,
    // New memory write methods
    updateProfile,
    addProjectNote,
    requestProjectNotes,
    requestActiveProject,
    setActiveProjectRequest,
    updateFact,
    updateProjectNote,
    deleteProjectNote,
  };
}
