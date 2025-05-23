// frontend/src/components/MarkdownDisplay.jsx
import React from 'react';
import ReactMarkdown from 'react-markdown';

const MarkdownDisplay = ({ markdownText }) => {
  if (!markdownText) {
    return (
      <div className="p-4 bg-gray-50 rounded-lg shadow">
        <p className="text-gray-500">No document processed yet, or document has no viewable content.</p>
      </div>
    );
  }

  return (
    // Apply some basic Tailwind typography for better Markdown rendering.
    // If you have @tailwindcss/typography, you can use `prose` or `prose-lg`.
    // For a basic version without the plugin:
    <div className="p-4 bg-white rounded-lg shadow space-y-3">
      <ReactMarkdown
        children={markdownText}
        components={{
          // Customize rendering of specific elements if needed
          // For example, add Tailwind classes to headings, paragraphs, lists etc.
          h1: ({node, ...props}) => <h1 className="text-2xl font-bold mb-2" {...props} />,
          h2: ({node, ...props}) => <h2 className="text-xl font-semibold mb-2" {...props} />,
          p: ({node, ...props}) => <p className="mb-2 leading-relaxed" {...props} />,
          ul: ({node, ...props})=> <ul className="list-disc list-inside mb-2" {...props} />,
          ol: ({node, ...props})=> <ol className="list-decimal list-inside mb-2" {...props} />,
          li: ({node, ...props})=> <li className="mb-1" {...props} />,
          code: ({node, inline, className, children, ...props}) => {
            const match = /language-(\w+)/.exec(className || '')
            return !inline && match ? (
              // For code blocks, you might want a proper syntax highlighter later
              <pre className="bg-gray-100 p-2 rounded overflow-x-auto text-sm" {...props}> 
                <code>{String(children).replace(/\n$/, '')}</code>
              </pre>
            ) : (
              <code className="bg-gray-200 px-1 rounded text-sm" {...props}>
                {children}
              </code>
            )
          }
        }}
      />
    </div>
  );
};

export default MarkdownDisplay;
