import ReactMarkdown from 'react-markdown';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
  if (!content) {
    return null;
  }

  return (
    <div className={`prose prose-invert prose-sm max-w-none ${className}`}>
      <ReactMarkdown
        components={{
          // Customize code blocks
          code({ node, inline, className, children, ...props }) {
            return inline ? (
              <code className="bg-bg-tertiary px-1 py-0.5 rounded text-accent-blue" {...props}>
                {children}
              </code>
            ) : (
              <pre className="bg-bg-tertiary p-3 rounded overflow-x-auto">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            );
          },
          // Customize links
          a({ node, children, href, ...props }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent-blue hover:underline"
                {...props}
              >
                {children}
              </a>
            );
          },
          // Customize headings
          h1({ node, children, ...props }) {
            return (
              <h1 className="text-lg font-bold text-text-primary mt-4 mb-2" {...props}>
                {children}
              </h1>
            );
          },
          h2({ node, children, ...props }) {
            return (
              <h2 className="text-base font-semibold text-text-primary mt-3 mb-2" {...props}>
                {children}
              </h2>
            );
          },
          h3({ node, children, ...props }) {
            return (
              <h3 className="text-sm font-semibold text-text-primary mt-2 mb-1" {...props}>
                {children}
              </h3>
            );
          },
          // Customize paragraphs
          p({ node, children, ...props }) {
            return (
              <p className="text-text-secondary mb-2 last:mb-0" {...props}>
                {children}
              </p>
            );
          },
          // Customize lists
          ul({ node, children, ...props }) {
            return (
              <ul className="list-disc list-inside text-text-secondary mb-2 space-y-1" {...props}>
                {children}
              </ul>
            );
          },
          ol({ node, children, ...props }) {
            return (
              <ol className="list-decimal list-inside text-text-secondary mb-2 space-y-1" {...props}>
                {children}
              </ol>
            );
          },
          // Customize blockquotes
          blockquote({ node, children, ...props }) {
            return (
              <blockquote
                className="border-l-2 border-accent-blue pl-3 my-2 text-text-muted italic"
                {...props}
              >
                {children}
              </blockquote>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
