import { MAP_VARIABLES, type Meta, type MapVarKey } from '../lib/types'

interface SidebarProps {
  ds: Meta
  timeIdx: number
  onTimeIdx: (i: number) => void
  depthIdx: number
  onDepthIdx: (i: number) => void
  varKey: MapVarKey
  onVarKey: (v: MapVarKey) => void
  lat: number
  lon: number
  onLat: (v: number) => void
  onLon: (v: number) => void
  onLand: boolean
  /** False on every tab but Maps: the whole panel slides out of view (not even a strip is left). */
  visible: boolean
  open: boolean
  onToggle: () => void
}

export function Sidebar({
  ds, timeIdx, onTimeIdx, depthIdx, onDepthIdx, varKey, onVarKey, lat, lon, onLat, onLon, onLand, visible, open, onToggle,
}: SidebarProps) {
  const selectedVar = MAP_VARIABLES.find((v) => v.key === varKey)!

  // One element for both states so the width can animate: the panel slides shut to a slim strip (and back).
  // The content keeps its full width while clipped, so it doesn't reflow mid-slide.
  return (
    <aside
      aria-hidden={!visible}
      className={`glass-panel relative shrink-0 overflow-hidden rounded-2xl transition-[width,margin,opacity] duration-500 ease-in-out ${
        !visible
          ? 'pointer-events-none w-full opacity-0 max-lg:hidden lg:-mr-3 lg:w-0 lg:border-0'
          : open
            ? 'w-full lg:w-72'
            : 'w-full lg:w-11'
      }`}
    >
      {!open && (
        <button
          onClick={onToggle}
          title="Show controls"
          className="text-xs font-semibold uppercase tracking-wider text-surf-400 hover:bg-white/10 max-lg:block max-lg:w-full max-lg:py-3 lg:absolute lg:inset-0 lg:[writing-mode:vertical-rl]"
        >
          Controls &raquo;
        </button>
      )}
      <div
        className={`scrollbar-thin flex h-full flex-col gap-6 overflow-y-auto p-5 transition-opacity duration-200 lg:w-72 ${
          open ? 'opacity-100' : 'pointer-events-none opacity-0 max-lg:hidden'
        }`}
      >
      <div className="flex items-center justify-between">
        <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-surf-400">Controls</h2>
        <button
          onClick={onToggle}
          title="Hide controls"
          className="-mt-4 rounded-lg px-2 py-1 text-sm text-slate-200 hover:bg-white/10 hover:text-slate-100"
        >
          &laquo;
        </button>
      </div>

      <Field label={`Date — ${ds.times[timeIdx]}`}>
        <input
          type="range"
          min={0}
          max={ds.times.length - 1}
          value={timeIdx}
          onChange={(e) => onTimeIdx(Number(e.target.value))}
          className="range-slider"
        />
      </Field>

      <Field label={`Depth — ${ds.depths[depthIdx]} m`} disabled={selectedVar.depthFree}>
        <input
          type="range"
          min={0}
          max={ds.depths.length - 1}
          value={depthIdx}
          onChange={(e) => onDepthIdx(Number(e.target.value))}
          disabled={selectedVar.depthFree}
          className="range-slider"
        />
        {selectedVar.depthFree && (
          <p className="mt-1 text-xs text-slate-300">This variable is 2-D — depth does not apply.</p>
        )}
      </Field>

      <Field label="Map variable">
        <select
          value={varKey}
          onChange={(e) => onVarKey(e.target.value as MapVarKey)}
          className="w-full rounded-lg border border-abyss-500 bg-abyss-900 px-3 py-2 text-sm text-slate-200 outline-none focus:border-surf-500"
        >
          {MAP_VARIABLES.map((v) => (
            <option key={v.key} value={v.key}>
              {v.label}
            </option>
          ))}
        </select>
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field label="Latitude (N)">
          <input
            type="number"
            step={0.25}
            min={5}
            max={30}
            value={lat}
            onChange={(e) => onLat(Number(e.target.value))}
            className="w-full rounded-lg border border-abyss-500 bg-abyss-900 px-3 py-2 text-sm text-slate-200 outline-none focus:border-surf-500"
          />
        </Field>
        <Field label="Longitude (E)">
          <input
            type="number"
            step={0.25}
            min={45}
            max={105}
            value={lon}
            onChange={(e) => onLon(Number(e.target.value))}
            className="w-full rounded-lg border border-abyss-500 bg-abyss-900 px-3 py-2 text-sm text-slate-200 outline-none focus:border-surf-500"
          />
        </Field>
      </div>

      {!onLand ? null : (
        <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-400">
          That point is on land — pick an ocean location for profiles/time series.
        </p>
      )}
      </div>
    </aside>
  )
}

function Field({ label, children, disabled }: { label: string; children: React.ReactNode; disabled?: boolean }) {
  return (
    <label className={`block ${disabled ? 'opacity-40' : ''}`}>
      <span className="mb-2 block text-sm font-medium text-slate-300">{label}</span>
      {children}
    </label>
  )
}
