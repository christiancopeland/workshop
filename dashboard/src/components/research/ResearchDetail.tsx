import { useCallback } from 'react';
import { ResearchDetail as ResearchDetailType, ResearchSource } from '../../types';
import { formatRelativeTime } from '../../utils';

interface ResearchDetailProps {
  platform: ResearchDetailType;
}

export function ResearchDetail({ platform }: ResearchDetailProps) {
  const handleCopyMarkdown = useCallback(() => {
    const markdown = generateMarkdown(platform);
    navigator.clipboard.writeText(markdown);
  }, [platform]);

  const sources = platform.sources || [];
  const keyPoints = platform.key_points || [];

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold text-text-primary">
            {platform.topic || 'Research'}
          </h2>
          <div className="text-xs text-text-muted mt-1">
            {platform.created_at && `Created ${formatRelativeTime(platform.created_at)}`}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleCopyMarkdown} className="btn btn-ghost btn-sm">
            📋 Copy Markdown
          </button>
        </div>
      </div>

      {/* Original Query */}
      {platform.original_query && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-text-primary mb-2">Original Query</h3>
          <p className="text-sm text-text-secondary bg-bg-secondary border border-border rounded-lg p-4">
            {platform.original_query}
          </p>
        </div>
      )}

      {/* Summary */}
      {platform.summary && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-text-primary mb-2">Summary</h3>
          <p className="text-sm text-text-secondary bg-bg-secondary border border-border rounded-lg p-4">
            {platform.summary}
          </p>
        </div>
      )}

      {/* Key Points */}
      {keyPoints.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-text-primary mb-2">
            Key Points
          </h3>
          <ul className="space-y-2">
            {keyPoints.map((point: string, i: number) => (
              <li
                key={i}
                className="flex items-start gap-2 text-sm text-text-secondary"
              >
                <span className="text-accent-blue shrink-0">•</span>
                <span>{point}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Sources */}
      <div>
        <h3 className="text-sm font-semibold text-text-primary mb-2">
          Sources ({sources.length})
        </h3>
        <div className="space-y-2">
          {sources.map((source: ResearchSource, i: number) => (
            <div
              key={i}
              className="bg-bg-secondary border border-border rounded-lg p-3"
            >
              <a
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-medium text-accent-blue hover:underline"
              >
                {source.title || source.url}
              </a>
              {source.snippet && (
                <p className="text-xs text-text-muted mt-1 line-clamp-2">
                  {source.snippet}
                </p>
              )}
              <div className="text-xs text-text-muted mt-1 truncate">
                {source.url}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function generateMarkdown(platform: ResearchDetailType): string {
  let md = `# ${platform.topic || 'Research'}\n\n`;

  if (platform.original_query) {
    md += `## Query\n\n${platform.original_query}\n\n`;
  }

  if (platform.summary) {
    md += `## Summary\n\n${platform.summary}\n\n`;
  }

  const keyPoints = platform.key_points || [];
  if (keyPoints.length > 0) {
    md += `## Key Points\n\n`;
    keyPoints.forEach((point: string) => {
      md += `- ${point}\n`;
    });
    md += '\n';
  }

  const sources = platform.sources || [];
  md += `## Sources\n\n`;
  sources.forEach((source: ResearchSource) => {
    md += `- [${source.title || source.url}](${source.url})\n`;
    if (source.snippet) {
      md += `  > ${source.snippet}\n`;
    }
  });

  return md;
}
