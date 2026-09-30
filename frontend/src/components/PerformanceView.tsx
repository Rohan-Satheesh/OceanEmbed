export function PerformanceView() {
  return (
    <div className="glass-panel rounded-2xl p-8 text-center">
      <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-surf-500/10 text-surf-400">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 3v18h18" strokeLinecap="round" />
          <path d="M7 15l4-6 3 3 5-8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <p className="text-sm text-slate-300">No evaluation report yet.</p>
      <p className="mt-1 text-xs text-slate-300">
        Run <code className="rounded bg-abyss-900 px-1.5 py-0.5 text-surf-400">python -m evaluation.report</code> after the
        training experiments finish (Chunks 4-6) — this panel will render the RMSE-by-depth and reliability plots once
        the report exists.
      </p>
    </div>
  )
}
