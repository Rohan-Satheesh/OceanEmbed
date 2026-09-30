import { useEffect } from 'react'

interface ModalProps {
  title: string
  subtitle?: string
  onClose: () => void
  children: React.ReactNode
}

export function Modal({ title, subtitle, onClose, children }: ModalProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-abyss-800/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-sand-400/70 bg-sand-300 shadow-2xl">
        <div className="flex items-start justify-between border-b border-sand-400/70 px-5 py-4">
          <div>
            <h2 className="text-base font-bold text-abyss-900 dark:text-slate-100">{title}</h2>
            {subtitle && <p className="mt-0.5 text-xs text-abyss-600 dark:text-slate-400">{subtitle}</p>}
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-abyss-700 hover:bg-abyss-700/10 hover:text-abyss-950 dark:text-slate-300 dark:hover:bg-white/10 dark:hover:text-white"
          >
            ✕
          </button>
        </div>
        <div className="scrollbar-thin overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  )
}
