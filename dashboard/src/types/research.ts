export interface ResearchSource {
  title: string;
  url: string;
  snippet?: string;
  domain?: string;
}

// Shape returned by server from _get_research_library
export interface ResearchPlatform {
  filename: string;
  topic: string;              // Was platform_name
  original_query: string;     // Was summary
  source_count: number;       // Was sources array
  created_at: string;
  size_kb: number;
  is_active: boolean;
}

// Full research detail (returned by _get_research_detail)
// This is the actual content of the JSON file
export interface ResearchDetail {
  topic?: string;
  original_query?: string;
  sources?: ResearchSource[];
  created_at?: string;
  key_points?: string[];
  summary?: string;
  // Allow any other fields from the JSON
  [key: string]: unknown;
}

export interface ResearchLibrary {
  platforms: ResearchPlatform[];
  active_count: number;
  total_count: number;
}
