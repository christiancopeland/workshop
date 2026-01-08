export interface MemoryStats {
  semantic_count: number;
  facts_count: number;
  projects_count: number;
  profile_exists: boolean;
  message_count: number;
}

export interface Fact {
  key: string;
  value: string;
  category?: string;
  created_at: string;
  updated_at?: string;
}

export interface Project {
  name: string;
  path: string;
  is_active: boolean;
  last_accessed?: string;
}

export interface MemorySearchResult {
  content: string;
  metadata: Record<string, unknown>;
  distance: number;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

export interface MemoryState {
  stats: MemoryStats | null;
  facts: Fact[];
  projects: Project[];
  profile: string;
  messages: Message[];
  searchResults: MemorySearchResult[];
  searchQuery: string;
}

export interface ProjectNote {
  id: number;
  content: string;
  category: string;
  created_at: string;
}

export interface KnowledgeItem {
  id: string;
  type: 'fact' | 'note' | 'memory';
  title?: string;
  content: string;
  timestamp: string;
  category?: string;
  metadata?: Record<string, unknown>;
}

export interface ActiveProject {
  id: number;
  name: string;
  path: string | null;
  description: string | null;
  last_active: string;
  created_at: string;
}
