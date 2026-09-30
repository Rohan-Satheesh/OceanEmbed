import { useTheme } from '../../lib/theme'
import GradientWaves from './GradientWaves'

// Fixed, non-interactive shoreline waves behind the whole app. Horizon fades into the sand page
// color; wave bodies are turquoise with white foam crests.
export function WaveBackground() {
  const [theme] = useTheme()
  const dark = theme === 'dark'
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10">
      <GradientWaves
        key={theme}
        horizonColor={dark ? '#081a24' : '#eed9a8'}
        waveColor={dark ? '#0c4a5e' : '#19a7b8'}
        crestColor={dark ? '#6fd6e0' : '#ffffff'}
        speed={0.35}
        amplitude={3}
        fogDepth={90}
        tilt={1.0}
        zoom={1.3}
        detail="medium"
        mouseInteraction={false}
        grain={false}
      />
    </div>
  )
}
