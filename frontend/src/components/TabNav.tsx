const TABS = ['Maps', 'Vertical profile', 'Time series', 'Regional summary', 'Model performance', 'About'] as const
export type Tab = (typeof TABS)[number]

interface TabNavProps {
  active: Tab
  onChange: (t: Tab) => void
}

/** Top-bar navigation links; sits inside the Navbar. */
export function TabNav({ active, onChange }: TabNavProps) {
  return (
    <nav className="scrollbar-thin flex min-w-0 gap-1 overflow-x-auto">
      {TABS.map((tab) => (
        <button
          key={tab}
          onClick={() => onChange(tab)}
          className={`shrink-0 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors ${
            active === tab
              ? 'bg-surf-500/15 text-surf-400 shadow-[inset_0_0_0_1px_rgba(31,201,194,0.35)]'
              : 'text-slate-200 hover:bg-white/10 hover:text-white'
          }`}
        >
          {tab}
        </button>
      ))}
    </nav>
  )
}
