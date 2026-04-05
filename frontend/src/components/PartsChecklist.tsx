import { useEffect, useState } from 'react';
import type { Part } from '../api/legogen';

interface PartsChecklistProps {
  parts: Part[];
  highlightPartId?: string;
}

export default function PartsChecklist({ parts, highlightPartId }: PartsChecklistProps) {
  const [checked, setChecked] = useState<Set<string>>(new Set());

  useEffect(() => {
    setChecked(new Set());
  }, [parts]);

  const toggle = (partId: string) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(partId)) next.delete(partId);
      else next.add(partId);
      return next;
    });
  };

  if (!parts.length) {
    return <p className="text-xs text-gray-600 italic">No parts for this step.</p>;
  }

  return (
    <div className="space-y-1">
      <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Parts Checklist</h4>
      {parts.map((part) => {
        const isHighlighted = part.part_id === highlightPartId;
        const isChecked = checked.has(part.part_id);
        return (
          <label
            key={part.part_id}
            className={`flex items-center gap-2.5 px-2 py-1.5 rounded-lg cursor-pointer transition-colors ${
              isHighlighted
                ? 'bg-green-500/10 border border-green-500/20'
                : 'hover:bg-white/5'
            } ${isChecked ? 'opacity-50' : ''}`}
          >
            <input
              type="checkbox"
              checked={isChecked}
              onChange={() => toggle(part.part_id)}
              className="w-3.5 h-3.5 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500/30 focus:ring-offset-0"
            />
            <span
              className="w-3.5 h-3.5 rounded-sm flex-shrink-0 border border-white/10"
              style={{ backgroundColor: part.color_hex }}
              title={part.color}
            />
            <span className={`text-xs flex-grow ${isChecked ? 'line-through text-gray-600' : 'text-gray-300'}`}>
              {part.name}
            </span>
            <span className="text-xs text-gray-500 font-mono">x{part.quantity}</span>
          </label>
        );
      })}
    </div>
  );
}
