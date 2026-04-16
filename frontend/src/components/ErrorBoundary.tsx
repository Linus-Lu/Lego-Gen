import React from 'react';

interface ErrorBoundaryProps { children: React.ReactNode; }
interface ErrorBoundaryState { hasError: boolean; error: Error | null; }

export default class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  handleReset = () => this.setState({ hasError: false, error: null });

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div className="min-h-screen bp-grid grid place-items-center p-8">
        <div className="max-w-md w-full bp-frame border border-[var(--color-line)] bg-[var(--color-ink-2)] p-8">
          <p className="label-accent mb-3">ERR // RUNTIME HALT</p>
          <p className="display text-3xl text-[var(--color-fg-strong)] mb-2">Something broke.</p>
          <p className="mono text-xs text-[var(--color-dim)] break-words mb-6">
            {this.state.error?.message ?? 'An unexpected error occurred while rendering.'}
          </p>
          <button onClick={this.handleReset} className="btn-primary">↻ Reset view</button>
        </div>
      </div>
    );
  }
}
