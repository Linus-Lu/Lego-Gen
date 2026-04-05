interface MetricsBarProps {
  label: string;
  value: number;
  max: number;
  color?: string;
}

export default function MetricsBar({ label, value, max, color = '#3b82f6' }: MetricsBarProps) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300 font-mono">{typeof value === 'number' ? value.toFixed(value % 1 ? 2 : 0) : value}</span>
      </div>
      <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}
