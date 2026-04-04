import React, { useState } from 'react';
import type { ValidationReport, CheckResult } from '../api/legogen';

interface ValidationPanelProps {
  validation: ValidationReport | null | undefined;
}

function statusIcon(status: 'pass' | 'warn' | 'fail') {
  if (status === 'pass') {
    return (
      <svg className="w-4 h-4 text-green-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
      </svg>
    );
  }
  if (status === 'warn') {
    return (
      <svg className="w-4 h-4 text-yellow-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
      </svg>
    );
  }
  return (
    <svg className="w-4 h-4 text-red-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

function formatName(name: string) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function scoreColor(score: number) {
  if (score >= 80) return { bg: 'bg-green-500/15', border: 'border-green-500/30', text: 'text-green-400', ring: 'ring-green-500/20' };
  if (score >= 50) return { bg: 'bg-yellow-500/15', border: 'border-yellow-500/30', text: 'text-yellow-400', ring: 'ring-yellow-500/20' };
  return { bg: 'bg-red-500/15', border: 'border-red-500/30', text: 'text-red-400', ring: 'ring-red-500/20' };
}

function CheckSection({ title, checks }: { title: string; checks: CheckResult[] }) {
  const [expanded, setExpanded] = useState(true);
  const hasIssues = checks.some((c) => c.status !== 'pass');

  return (
    <div className="space-y-1">
      <button
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className="flex items-center gap-2 w-full text-left text-xs font-semibold text-gray-400 uppercase tracking-wider py-1 hover:text-gray-300 transition-colors"
      >
        <svg
          className={`w-3 h-3 transition-transform ${expanded ? 'rotate-90' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2.5}
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        </svg>
        {title}
        {hasIssues && (
          <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/10 text-yellow-400 normal-case tracking-normal font-normal">
            {checks.filter((c) => c.status !== 'pass').length} issue(s)
          </span>
        )}
      </button>

      {expanded && (
        <div className="space-y-1 pl-1">
          {checks.map((check) => (
            <CheckRow key={check.name} check={check} />
          ))}
        </div>
      )}
    </div>
  );
}

function CheckRow({ check }: { check: CheckResult }) {
  const [detailsOpen, setDetailsOpen] = useState(check.status !== 'pass');

  return (
    <div
      className={`rounded-lg px-2.5 py-2 text-xs ${
        check.status === 'pass'
          ? 'bg-white/[0.02]'
          : check.status === 'warn'
          ? 'bg-yellow-500/[0.04] border border-yellow-500/10'
          : 'bg-red-500/[0.04] border border-red-500/10'
      }`}
    >
      <button
        onClick={() => setDetailsOpen(!detailsOpen)}
        aria-expanded={detailsOpen}
        aria-label={`${formatName(check.name)}: ${check.status}`}
        className="flex items-center gap-2 w-full text-left"
      >
        {statusIcon(check.status)}
        <span className="text-gray-300 font-medium">{formatName(check.name)}</span>
      </button>
      {detailsOpen && (
        <p className="mt-1 ml-6 text-gray-500 leading-relaxed">{check.message}</p>
      )}
    </div>
  );
}

const ValidationPanel: React.FC<ValidationPanelProps> = ({ validation }) => {
  if (!validation) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600 text-sm">
        No validation data available.
      </div>
    );
  }

  const sc = scoreColor(validation.score);
  const legalityChecks = validation.checks.filter((c) => c.category === 'legality');
  const stabilityChecks = validation.checks.filter((c) => c.category === 'stability');

  return (
    <div className="flex flex-col gap-3 p-1 h-full overflow-y-auto">
      {/* Score badge */}
      <div className="flex items-center gap-3">
        <div
          className={`flex items-center justify-center w-14 h-14 rounded-xl ${sc.bg} ${sc.border} border ring-1 ${sc.ring}`}
        >
          <span className={`text-xl font-bold ${sc.text}`}>{validation.score}</span>
        </div>
        <div>
          <p className={`text-sm font-semibold ${sc.text}`}>
            {validation.score >= 80 ? 'Stable Build' : validation.score >= 50 ? 'Some Issues' : 'Unstable Build'}
          </p>
          <p className="text-[11px] text-gray-500">{validation.summary}</p>
        </div>
      </div>

      {/* Check sections */}
      {legalityChecks.length > 0 && (
        <CheckSection title="Legality" checks={legalityChecks} />
      )}
      {stabilityChecks.length > 0 && (
        <CheckSection title="Stability" checks={stabilityChecks} />
      )}
    </div>
  );
};

export default ValidationPanel;
