import LatticeLoader from './loading/LatticeLoader'

/** The lattice loader, tuned for the dark glass panels (light ink). */
export function PanelLoader({ label = 'Loading', color = '#e2e8f0' }: { label?: string; color?: string }) {
  return (
    <LatticeLoader
      label={label}
      showTimer={false}
      pattern="orbit"
      grid={3}
      shape="round"
      color={color}
      cellSize={6}
      gap={2}
      fontSize={14}
    />
  )
}

export function LoadingPanel({ label }: { label?: string }) {
  return (
    <div className="glass-panel flex items-center justify-center rounded-2xl p-10 font-medium text-slate-100">
      <PanelLoader label={label} />
    </div>
  )
}

export function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-coral-600/40 bg-coral-600/10 p-6 text-sm text-coral-500">
      Could not load data from the backend: {message}
    </div>
  )
}
