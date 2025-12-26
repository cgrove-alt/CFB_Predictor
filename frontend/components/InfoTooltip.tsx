'use client';

import { useState } from 'react';

interface InfoTooltipProps {
  content: React.ReactNode;
  className?: string;
}

export default function InfoTooltip({ content, className = '' }: InfoTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className={`relative inline-block ${className}`}>
      {/* Info icon button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        className="ml-1 inline-flex items-center justify-center w-4 h-4 text-xs font-bold text-gray-400 hover:text-blue-400 rounded-full border border-gray-500 hover:border-blue-400 transition-colors"
        aria-label="More info"
      >
        ?
      </button>

      {/* Tooltip content */}
      {isOpen && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3 bg-gray-800 border border-gray-600 rounded-lg shadow-xl text-sm text-gray-200">
          {content}
          {/* Arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-gray-800" />
        </div>
      )}
    </div>
  );
}

// Pre-defined tooltip content for Edge explanation
export function EdgeTooltip() {
  return (
    <InfoTooltip
      content={
        <div className="space-y-2">
          <p className="font-semibold text-blue-400">EDGE: Predicted advantage over Vegas</p>
          <p className="text-gray-300">
            <span className="font-mono">Edge = Predicted Margin - Vegas Spread</span>
          </p>
          <p className="text-gray-400 text-xs">
            Example: Vegas says -7, model predicts -10 → Edge = +3 pts
          </p>
          <ul className="text-xs text-gray-400 space-y-1 mt-2">
            <li>• <span className="text-green-400">Positive</span> = Favors your pick</li>
            <li>• Larger edge = Higher confidence</li>
            <li>• Combined with cover probability for bet sizing</li>
          </ul>
        </div>
      }
    />
  );
}
