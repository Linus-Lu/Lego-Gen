import { useState } from 'react';
import type { Part } from '../api/legogen';

interface ColorLegendProps {
  dominantColors: string[];
  allParts: Part[];
}

const ColorLegend: React.FC<ColorLegendProps> = ({ dominantColors, allParts }) => {
  const [expanded, setExpanded] = useState(false);

  // Build color summary: hex -> { name, count, is_trans }
  const colorMap = new Map<string, { name: string; count: number; is_trans: boolean }>();
  for (const part of allParts) {
    const existing = colorMap.get(part.color_hex);
    if (existing) {
      existing.count += part.quantity;
    } else {
      colorMap.set(part.color_hex, {
        name: part.color,
        count: part.quantity,
        is_trans: part.is_trans ?? false,
      });
    }
  }

  const allColors = [...colorMap.entries()].sort((a, b) => b[1].count - a[1].count);

  return (
    <div className="border-t border-gray-700 pt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full text-[10px] uppercase tracking-wider text-gray-500 font-semibold hover:text-gray-300 transition px-1"
      >
        <span>Color Palette ({allColors.length})</span>
        <span>{expanded ? '▾' : '▸'}</span>
      </button>

      {/* Dominant colors — always visible */}
      <div className="flex gap-2 mt-2 px-1">
        {dominantColors.map((name, i) => {
          const entry = allColors.find(([_, v]) => v.name === name);
          const hex = entry?.[0] ?? '#888';
          return (
            <div key={i} className="flex items-center gap-1.5">
              <div
                className="w-4 h-4 rounded border border-white/20"
                style={{ backgroundColor: hex }}
              />
              <span className="text-[10px] text-gray-400">{name}</span>
            </div>
          );
        })}
      </div>

      {/* Expanded: all colors */}
      {expanded && (
        <div className="grid grid-cols-2 gap-1 mt-2 px-1 max-h-32 overflow-y-auto">
          {allColors.map(([hex, info]) => (
            <div key={hex} className="flex items-center gap-1.5 py-0.5">
              <div
                className={`w-3 h-3 rounded border border-white/15 flex-shrink-0 ${
                  info.is_trans ? 'opacity-60' : ''
                }`}
                style={{ backgroundColor: hex }}
              />
              <span className="text-[10px] text-gray-400 truncate">{info.name}</span>
              <span className="text-[10px] text-gray-600 ml-auto">{info.count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ColorLegend;
