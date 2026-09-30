import { useEffect, useRef, useState } from 'react'
import { Logo } from './Logo'
import { useTheme } from '../lib/theme'
import LatticeLoader from './loading/LatticeLoader'

const TIPS = [
  'Diving past the thermocline...',
  'Untangling kelp from the data pipeline...',
  'Bribing ARGO floats to profile faster...',
  'Convincing eddies to hold still...',
  'Counting fish in the Bay of Bengal...',
  'Waking up the mixed layer...',
  'Whale-approved satellite data...',
  'Teaching neurons to swim...',
  'Reticulating isotherms...',
  'Asking GLORYS nicely for its temperature...',
  'Charting the twenty-degree isotherm...',
  'Polishing the thermocline...',
  "Don't feed the sharks the training data...",
  'Currents currently loading...',
  'Salinity levels: nominal.',
]

const TOTAL_MS = 2600
const TIP_INTERVAL_MS = 850
const FADE_MS = 900

interface LoadingScreenProps {
  /** True once the data source has connected; the screen never finishes before this. */
  dataReady: boolean
  onComplete: () => void
}

export function LoadingScreen({ dataReady, onComplete }: LoadingScreenProps) {
  const [theme] = useTheme()
  const [tipIndex, setTipIndex] = useState(() => Math.floor(Math.random() * TIPS.length))
  const [fadeOut, setFadeOut] = useState(false)
  const [shown, setShown] = useState(false)
  const [minElapsed, setMinElapsed] = useState(false)
  const finishedRef = useRef(false)

  useEffect(() => {
    const raf = requestAnimationFrame(() => setShown(true))
    const tipTimer = setInterval(() => {
      setTipIndex((i) => (i + 1) % TIPS.length)
    }, TIP_INTERVAL_MS)
    const minTimer = setTimeout(() => setMinElapsed(true), TOTAL_MS)

    return () => {
      cancelAnimationFrame(raf)
      clearInterval(tipTimer)
      clearTimeout(minTimer)
    }
  }, [])

  // Exit once BOTH the minimum display time has passed and the data source has connected.
  useEffect(() => {
    if (!minElapsed || !dataReady || finishedRef.current) return
    finishedRef.current = true
    setFadeOut(true)
    const t = setTimeout(onComplete, FADE_MS)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minElapsed, dataReady])

  return (
    <div
      className={`fixed inset-0 z-[100] flex flex-col items-center justify-center gap-5 transition-opacity ${
        fadeOut || !shown ? 'pointer-events-none opacity-0' : 'opacity-100'
      }`}
      style={{ transitionDuration: `${FADE_MS}ms` }}
    >
      <div className="relative flex flex-col items-center gap-5 rounded-3xl border border-white/40 bg-white/30 px-10 py-8 shadow-2xl backdrop-blur-xl backdrop-saturate-150 dark:border-white/20 dark:bg-abyss-800/40">
        <div className="flex items-center gap-3">
          <Logo size={56} />
          <div>
            <div className="text-xl font-bold text-abyss-950 dark:text-white">OceanEmbed</div>
            <div className="text-sm font-medium text-abyss-800 dark:text-slate-300">Subsurface Ocean Temperature Reconstruction</div>
          </div>
        </div>

        <div className="font-semibold text-[#083a6b] dark:text-surf-400">
          <LatticeLoader
            label={TIPS[tipIndex]}
            showTimer={false}
            pattern="orbit"
            grid={3}
            shape="round"
            color={theme === 'dark' ? '#7fe3ea' : '#083a6b'}
            doneColor="#38e3d8"
            cellSize={7}
            gap={2}
            fontSize={14}
          />
        </div>
      </div>
    </div>
  )
}
