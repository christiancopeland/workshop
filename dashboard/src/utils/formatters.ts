import { formatDistanceToNow, format } from 'date-fns';

export function formatTimestamp(timestamp: string): string {
  // Backend sends time_str in format "HH:mm:ss.SSS" - if it's already in this format, return it
  if (/^\d{2}:\d{2}:\d{2}/.test(timestamp)) {
    return timestamp;
  }
  // Otherwise try to parse as a full date
  const date = new Date(timestamp);
  if (isNaN(date.getTime())) {
    return timestamp; // Return as-is if unparseable
  }
  return format(date, 'HH:mm:ss.SSS');
}

export function formatRelativeTime(timestamp: string | undefined | null): string {
  if (!timestamp) return 'unknown';
  const date = new Date(timestamp);
  if (isNaN(date.getTime())) return 'unknown';
  return formatDistanceToNow(date, { addSuffix: true });
}

export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  } else {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function formatVRAM(used: number, total: number): string {
  return `${formatBytes(used)} / ${formatBytes(total)}`;
}

export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - 3) + '...';
}

export function formatJSON(obj: unknown): string {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

export function getEventSummary(type: string, data: Record<string, unknown>): string {
  switch (type) {
    case 'user_input':
      return truncate(String(data.message || data.input || ''), 100);
    case 'assistant_response':
      return truncate(String(data.response || data.message || ''), 100);
    case 'tool_calling':
      return `Calling ${data.tool_name || 'tool'}`;
    case 'tool_result':
      return `${data.tool_name || 'Tool'} completed`;
    case 'tool_error':
      return `${data.tool_name || 'Tool'} failed: ${truncate(String(data.error || ''), 50)}`;
    case 'llm_calling': {
      const model = data.model || 'LLM';
      const msgCount = data.message_count as number | undefined;
      return msgCount ? `Calling ${model} (${msgCount} msgs)` : `Calling ${model}`;
    }
    case 'llm_complete':
      return `${data.model || 'LLM'} responded`;
    case 'skill_matched':
      return `Matched skill: ${data.skill_name || 'unknown'}`;
    case 'intent_detected':
      return `Intent: ${data.intent || 'unknown'}`;
    case 'research_started':
      return `Research: ${truncate(String(data.topic || data.query || ''), 50)}`;
    case 'research_complete':
      return `Research completed: ${data.sources_count || 0} sources`;
    case 'task_started':
      return `Started: ${truncate(String(data.task || ''), 50)}`;
    case 'task_completed':
      return `Completed: ${truncate(String(data.task || ''), 50)}`;
    case 'subagent_spawned':
      return `Spawned: ${data.model || 'subagent'}`;
    case 'model_swap':
      return `Swapped to ${data.new_model || 'model'}`;
    case 'error':
      return truncate(String(data.message || data.error || 'Unknown error'), 100);
    case 'warning':
      return truncate(String(data.message || 'Warning'), 100);
    case 'info':
      return truncate(String(data.message || 'Info'), 100);
    default:
      return type.replace(/_/g, ' ');
  }
}
