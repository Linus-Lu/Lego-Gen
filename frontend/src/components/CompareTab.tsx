import { useState, useEffect } from 'react';
import MetricsBar from './MetricsBar';

interface CheckpointResult {
  checkpoint: string;
  json_valid: boolean;
  parts_f1: number;
  color_f1: number;
  part_count: number;
}

interface ComparisonData {
  input: string;
  prompt: string;
  results: CheckpointResult[];
}

const SAMPLE_INPUTS = [
  'red_car',
  'house',
  'spaceship',
  'tree',
  'robot',
];

export default function CompareTab() {
  const [selectedInput, setSelectedInput] = useState(SAMPLE_INPUTS[0]);
  const [data, setData] = useState<ComparisonData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    async function loadComparison() {
      setLoading(true);
      setError('');
      try {
        const res = await fetch(`/data/comparisons/${selectedInput}.json`);
        if (!res.ok) throw new Error('Comparison data not found');
        const json = await res.json();
        setData(json);
      } catch {
        setData(null);
        setError('No comparison data available for this input. Run scripts/precompute_comparisons.py to generate.');
      } finally {
        setLoading(false);
      }
    }
    loadComparison();
  }, [selectedInput]);

  return (
    <div className="space-y-4">
      {/* Input selector */}
      <div className="flex items-center gap-3">
        <label className="text-xs text-gray-400">Test Input:</label>
        <select
          value={selectedInput}
          onChange={(e) => setSelectedInput(e.target.value)}
          className="bg-gray-800 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500/30"
        >
          {SAMPLE_INPUTS.map((inp) => (
            <option key={inp} value={inp}>
              {inp.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="flex items-center gap-2 py-8 justify-center">
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
        </div>
      )}

      {error && (
        <div className="text-center py-8">
          <p className="text-sm text-gray-500">{error}</p>
        </div>
      )}

      {data && !loading && (
        <>
          <p className="text-xs text-gray-500">Prompt: "{data.prompt}"</p>

          {/* Side-by-side panels */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {data.results.map((r) => (
              <div key={r.checkpoint} className="glass rounded-xl p-3 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-gray-300 truncate">{r.checkpoint}</h4>
                  <span className={`w-2 h-2 rounded-full ${r.json_valid ? 'bg-green-400' : 'bg-red-400'}`}
                    title={r.json_valid ? 'Valid JSON' : 'Invalid JSON'} />
                </div>

                <div className="space-y-2">
                  <MetricsBar label="Parts F1" value={r.parts_f1} max={1} color="#3b82f6" />
                  <MetricsBar label="Color F1" value={r.color_f1} max={1} color="#8b5cf6" />
                  <MetricsBar label="Part Count" value={r.part_count} max={200} color="#10b981" />
                </div>
              </div>
            ))}
          </div>

          {/* Summary bar chart */}
          <div className="glass rounded-xl p-4 space-y-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Summary — Parts F1 by Checkpoint</h4>
            {data.results.map((r) => (
              <MetricsBar
                key={r.checkpoint}
                label={r.checkpoint}
                value={r.parts_f1}
                max={1}
                color="#3b82f6"
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
