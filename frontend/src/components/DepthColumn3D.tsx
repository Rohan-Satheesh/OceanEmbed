import { Canvas } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import { useMemo, useRef, type ComponentRef, type ReactNode } from 'react'
import { colormap } from '../lib/colormaps'
import type { ProfilePoint } from '../lib/selectors'
import { useTheme } from '../lib/theme'

// Non-linear depth-to-scene-Y mapping (sqrt) so the thermocline (where most of the interesting
// structure is, ~50-300m) doesn't get crushed into a sliver next to the abyssal levels near 1000m.
const yForDepth = (depthM: number) => -Math.sqrt(depthM) * 0.85

// Beach palette: sand backdrop, turquoise sea surface, deep-blue ink.
const PALETTES = {
  light: { SAND: '#f6ecd2', INK: '#0b3a5b', SEA: '#19a7b8', TITLE: '#0b4f8a' },
  dark: { SAND: '#081a24', INK: '#d6eef2', SEA: '#1fc9c2', TITLE: '#7fe3ea' },
}

const PAN_STEP = 1.4

interface DepthColumn3DProps {
  profile: ProfilePoint[]
  lat: number
  lon: number
}

export function DepthColumn3D({ profile, lat, lon }: DepthColumn3DProps) {
  const [theme] = useTheme()
  const { SAND, INK, SEA, TITLE } = PALETTES[theme]
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null)

  const { min, max } = useMemo(() => {
    const temps = profile.map((p) => p.temp).filter(Number.isFinite)
    return { min: Math.min(...temps), max: Math.max(...temps) }
  }, [profile])

  const deepestY = yForDepth(profile[profile.length - 1]?.depthM ?? 1000)
  const startTarget: [number, number, number] = [0, deepestY / 2, 0]
  // Side-on and far enough back that the whole column (surface to 1000 m) fits in frame.
  const startCamera: [number, number, number] = [16, startTarget[1] + 3, 28]

  // Moves the camera and its look-at point together, so the view slides without changing the angle or zoom.
  function slide(dy: number) {
    const c = controls.current
    if (!c) return
    const nextY = Math.min(2, Math.max(deepestY - 1, c.target.y + dy))
    const applied = nextY - c.target.y
    c.target.y += applied
    c.object.position.y += applied
    c.update()
  }

  function resetView() {
    const c = controls.current
    if (!c) return
    c.target.set(...startTarget)
    c.object.position.set(...startCamera)
    c.update()
  }

  return (
    <div className="relative h-[460px] w-full overflow-hidden rounded-xl border border-sand-400/70" style={{ background: SAND }}>
      <Canvas camera={{ position: startCamera, fov: 42 }}>
        <color attach="background" args={[SAND]} />
        <fog attach="fog" args={[SAND, 45, 130]} />
        <ambientLight intensity={0.9} />
        <directionalLight position={[6, 8, 6]} intensity={1.0} />
        <directionalLight position={[-6, -4, -4]} intensity={0.35} />

        {/* Sea-surface plane for visual reference. */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
          <ringGeometry args={[0, 4.2, 64]} />
          <meshBasicMaterial color={SEA} transparent opacity={0.28} />
        </mesh>

        {/* Column spine connecting every depth disk. */}
        <mesh position={[0, deepestY / 2, 0]}>
          <cylinderGeometry args={[0.03, 0.03, -deepestY, 8]} />
          <meshBasicMaterial color={INK} />
        </mesh>

        {profile.map((p, i) => {
          if (!Number.isFinite(p.temp)) return null // below the local seafloor: no disk
          const t = (p.temp - min) / (max - min || 1)
          const color = colormap('turbo', t)
          const y = yForDepth(p.depthM)
          const radius = 2.6 - (p.depthM / (profile[profile.length - 1]?.depthM || 1000)) * 1.1
          return (
            <group key={i} position={[0, y, 0]}>
              <mesh>
                <cylinderGeometry args={[radius, radius, 0.16, 56]} />
                <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.12} />
              </mesh>
              <Text position={[radius + 0.5, 0, 0]} fontSize={0.3} color={INK} anchorX="left" anchorY="middle">
                {`${p.depthM} m  —  ${p.temp.toFixed(2)} °C  (±${(1.96 * p.std).toFixed(2)})`}
              </Text>
            </group>
          )
        })}

        <Text position={[0, 1.4, 0]} fontSize={0.4} color={TITLE} anchorX="center" anchorY="middle">
          {`${lat.toFixed(2)}N ${lon.toFixed(2)}E`}
        </Text>

        <OrbitControls
          ref={controls}
          target={startTarget}
          zoomToCursor
          minDistance={3}
          maxDistance={60}
        />
      </Canvas>

      {/* Vertical navigation: pan up/down the column (also works after zooming in). */}
      <div className="absolute right-3 top-3 flex flex-col gap-1.5">
        <NavButton label="Move up" onClick={() => slide(PAN_STEP)}>
          ▲
        </NavButton>
        <NavButton label="Move down" onClick={() => slide(-PAN_STEP)}>
          ▼
        </NavButton>
        <NavButton label="Reset view" onClick={resetView}>
          ⟲
        </NavButton>
      </div>
    </div>
  )
}

function NavButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      title={label}
      aria-label={label}
      className="flex h-9 w-9 items-center justify-center rounded-lg border border-abyss-700/30 bg-abyss-700 text-sm text-sand-300 shadow-md transition-colors hover:bg-abyss-600"
    >
      {children}
    </button>
  )
}
