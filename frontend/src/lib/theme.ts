import { useSyncExternalStore } from 'react'

export type Theme = 'light' | 'dark'

const KEY = 'oceanembed-theme'
const listeners = new Set<() => void>()

function read(): Theme {
  try {
    const saved = localStorage.getItem(KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch { /* storage blocked — fall through to the OS preference */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

let current: Theme = read()

function apply(t: Theme) {
  document.documentElement.classList.toggle('dark', t === 'dark')
  document.documentElement.style.colorScheme = t
}

/** Call once before first render so the page never flashes the wrong theme. */
export function initTheme() {
  apply(current)
}

export function setTheme(t: Theme) {
  current = t
  apply(t)
  try { localStorage.setItem(KEY, t) } catch { /* ignore */ }
  listeners.forEach((l) => l())
}

export function useTheme(): [Theme, () => void] {
  const theme = useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => listeners.delete(cb) },
    () => current,
  )
  return [theme, () => setTheme(theme === 'dark' ? 'light' : 'dark')]
}
