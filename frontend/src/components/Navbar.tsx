import { Logo } from './Logo'
import { ThemeToggle } from './ThemeToggle'
import { TabNav, type Tab } from './TabNav'

interface NavbarProps {
  active: Tab
  onChange: (t: Tab) => void
}

export function Navbar({ active, onChange }: NavbarProps) {
  return (
    <header className="shrink-0 border-b border-white/25 bg-abyss-800/65 shadow-lg backdrop-blur-xl backdrop-saturate-150">
      {/* Full width: logo pinned to the far left, tabs centred (the equal 1fr columns keep them centred). */}
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4 px-4 py-2 lg:px-6">
        <div className="flex items-center gap-2.5">
          <Logo size={34} />
          <span className="text-lg font-bold tracking-tight text-slate-100">OceanEmbed</span>
        </div>
        <TabNav active={active} onChange={onChange} />
        <div className="flex justify-end">
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
