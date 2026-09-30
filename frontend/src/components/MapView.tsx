import { useEffect, useMemo, useRef, useState } from 'react'
import { PanelLoader } from './AsyncPanels'
import { colormap, LAND_COLOR } from '../lib/colormaps'
import type { DataSource } from '../lib/dataSource'
import type { MapVariableDef } from '../lib/types'
import { useAsync } from '../lib/useAsync'

interface MapViewProps {
  src: DataSource
  varDef: MapVariableDef
  timeIdx: number
  depthIdx: number
  marker: { lat: number; lon: number }
  onCellClick: (lat: number, lon: number) => void
}

const MIN_ZOOM = 1
const MAX_ZOOM = 10
const GRID_VISIBLE_FROM_ZOOM = 2.5
const UPSAMPLE = 8 // canvas pixels per data cell (see the draw effect)
const LUT_SIZE = 512
const K3 = [1, 2, 1]

/** The outermost ring of the grid is known-wrong in the real SST/SSS/SSH and GLORYS fields (the regrid scales
 * partly-covered edge cells by their covered fraction; see CLAUDE.md), so its own values are never shown. Instead
 * each ring cell borrows its nearest interior neighbour, which keeps the ocean running right up to the map edge
 * (no dark border) until the pipeline is fixed. */
function inwardCell(cell: number, nLat: number, nLon: number): number {
  const i = Math.floor(cell / nLon)
  const j = cell - i * nLon
  return Math.min(nLat - 2, Math.max(1, i)) * nLon + Math.min(nLon - 2, Math.max(1, j))
}

function isRingCell(cell: number, nLat: number, nLon: number): boolean {
  const i = Math.floor(cell / nLon)
  const j = cell - i * nLon
  return i === 0 || j === 0 || i === nLat - 1 || j === nLon - 1
}

/** Ocean at this cell and a finite value at the cell it reads from (itself, or its interior neighbour on the ring). */
function isValidCell(cell: number, land: Uint8Array, field: Float32Array, nLat: number, nLon: number): boolean {
  const src = inwardCell(cell, nLat, nLon)
  return land[cell] === 0 && land[src] === 0 && Number.isFinite(field[src])
}

/** Separable blur with a normalised 1-D kernel; edges are clamped. */
function blurGrid(src: Float32Array, nLat: number, nLon: number, kernel: number[]): Float32Array {
  const r = (kernel.length - 1) / 2
  const norm = kernel.reduce((a, b) => a + b, 0)
  const tmp = new Float32Array(src.length)
  const out = new Float32Array(src.length)
  for (let i = 0; i < nLat; i++) {
    for (let j = 0; j < nLon; j++) {
      let acc = 0
      for (let k = -r; k <= r; k++) acc += kernel[k + r] * src[i * nLon + Math.min(nLon - 1, Math.max(0, j + k))]
      tmp[i * nLon + j] = acc / norm
    }
  }
  for (let i = 0; i < nLat; i++) {
    for (let j = 0; j < nLon; j++) {
      let acc = 0
      for (let k = -r; k <= r; k++) acc += kernel[k + r] * tmp[Math.min(nLat - 1, Math.max(0, i + k)) * nLon + j]
      out[i * nLon + j] = acc / norm
    }
  }
  return out
}

export function MapView({ src, varDef, timeIdx, depthIdx, marker, onCellClick }: MapViewProps) {
  const ds = src.meta
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const nLat = ds.lats.length
  const nLon = ds.lons.length

  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const dragState = useRef<{ startX: number; startY: number; panX: number; panY: number; moved: boolean } | null>(null)

  const { data: field, loading, error } = useAsync(
    () => src.mapField(varDef.key, timeIdx, depthIdx),
    [src, varDef.key, timeIdx, depthIdx],
  )
  const land = ds.land

  const { min, max, symmetric } = useMemo(() => {
    if (!field) return { min: 0, max: 1, symmetric: false }
    let lo = Infinity
    let hi = -Infinity
    for (let cell = 0; cell < field.length; cell++) {
      if (isRingCell(cell, nLat, nLon) || !isValidCell(cell, land, field, nLat, nLon)) continue
      const v = field[cell]
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    if (varDef.key === 'error') {
      const bound = Math.max(Math.abs(lo), Math.abs(hi)) || 1
      return { min: -bound, max: bound, symmetric: true }
    }
    if (!Number.isFinite(lo)) return { min: 0, max: 1, symmetric: false }
    return { min: lo, max: hi, symmetric: false }
  }, [field, land, nLat, nLon, varDef.key])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !field) return
    // The data are 0.25deg cells, so drawing one canvas pixel per cell (and letting the browser stretch it)
    // looks blocky, and a binary land mask gives staircase coastlines. Instead:
    //  1. smooth only the coast mask on the cell grid (3x3); values are left untouched, then
    //  2. render at UPSAMPLE x, bilinearly interpolating both, and anti-alias the coast where coverage crosses 0.5.
    const W = nLon * UPSAMPLE
    const H = nLat * UPSAMPLE
    canvas.width = W
    canvas.height = H
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(W, H)
    const out = img.data

    const lut = new Uint8Array(LUT_SIZE * 3)
    for (let k = 0; k < LUT_SIZE; k++) {
      const [r, g, b] = parseRgb(colormap(varDef.colormap, k / (LUT_SIZE - 1)))
      lut[k * 3] = r
      lut[k * 3 + 1] = g
      lut[k * 3 + 2] = b
    }
    const [lr, lg, lb] = hexToRgb(LAND_COLOR)
    const span = max - min || 1

    const valid = new Float32Array(nLat * nLon)
    for (let c = 0; c < valid.length; c++) valid[c] = isValidCell(c, land, field, nLat, nLon) ? 1 : 0
    const cover = blurGrid(valid, nLat, nLon, K3)
    const weighted = new Float32Array(valid.length)
    for (let c = 0; c < valid.length; c++) weighted[c] = valid[c] ? field[inwardCell(c, nLat, nLon)] : 0
    const num = blurGrid(weighted, nLat, nLon, K3)
    const den = blurGrid(valid, nLat, nLon, K3)
    const smooth = new Float32Array(valid.length)
    // Valid cells keep their own value (no blur); the blur only back-fills invalid neighbours so the colour
    // doesn't fade toward black at the coast.
    for (let c = 0; c < valid.length; c++) smooth[c] = valid[c] ? field[inwardCell(c, nLat, nLon)] : den[c] > 1e-6 ? num[c] / den[c] : 0

    for (let py = 0; py < H; py++) {
      const v = nLat - 1 - ((py + 0.5) / UPSAMPLE - 0.5) // row coordinate in cells, north at the top
      const i0 = Math.floor(v)
      const fy = v - i0
      const ia = Math.min(nLat - 1, Math.max(0, i0))
      const ib = Math.min(nLat - 1, Math.max(0, i0 + 1))
      for (let px = 0; px < W; px++) {
        const u = (px + 0.5) / UPSAMPLE - 0.5
        const j0 = Math.floor(u)
        const fx = u - j0
        const ja = Math.min(nLon - 1, Math.max(0, j0))
        const jb = Math.min(nLon - 1, Math.max(0, j0 + 1))

        const c00 = ia * nLon + ja
        const c01 = ia * nLon + jb
        const c10 = ib * nLon + ja
        const c11 = ib * nLon + jb
        const w00 = (1 - fx) * (1 - fy)
        const w01 = fx * (1 - fy)
        const w10 = (1 - fx) * fy
        const w11 = fx * fy

        const cov = w00 * cover[c00] + w01 * cover[c01] + w10 * cover[c10] + w11 * cover[c11]
        const o = (py * W + px) * 4
        out[o + 3] = 255
        if (cov <= 0.4) {
          out[o] = lr
          out[o + 1] = lg
          out[o + 2] = lb
          continue
        }
        const val = w00 * smooth[c00] + w01 * smooth[c01] + w10 * smooth[c10] + w11 * smooth[c11]
        const t = Math.min(1, Math.max(0, (val - min) / span))
        const k = Math.round(t * (LUT_SIZE - 1)) * 3
        const a = Math.min(1, (cov - 0.4) / 0.2) // anti-aliased coastline: 0.4 -> land, 0.6 -> ocean
        out[o] = lr + (lut[k] - lr) * a
        out[o + 1] = lg + (lut[k + 1] - lg) * a
        out[o + 2] = lb + (lut[k + 2] - lb) * a
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [field, land, nLat, nLon, min, max, varDef.colormap])

  const markerXPct = ((marker.lon - ds.lons[0]) / (ds.lons[nLon - 1] - ds.lons[0])) * 100
  const markerYPct = (1 - (marker.lat - ds.lats[0]) / (ds.lats[nLat - 1] - ds.lats[0])) * 100

  const depthLabel = varDef.depthFree ? '' : ` at ${ds.depths[depthIdx]} m`

  function clampPan(next: { x: number; y: number }, z: number) {
    // Keep at least a sliver of the map on-screen; overshoot allowed a bit for a natural feel.
    const bound = (z - 1) * 50 + 15
    return {
      x: Math.min(bound, Math.max(-bound, next.x)),
      y: Math.min(bound, Math.max(-bound, next.y)),
    }
  }

  function handlePointerDown(e: React.PointerEvent) {
    ;(e.target as Element).setPointerCapture(e.pointerId)
    dragState.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y, moved: false }
  }

  function handlePointerMove(e: React.PointerEvent) {
    const drag = dragState.current
    if (!drag) return
    const dx = e.clientX - drag.startX
    const dy = e.clientY - drag.startY
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) drag.moved = true
    const viewport = viewportRef.current
    const widthPct = viewport ? (dx / viewport.clientWidth) * 100 : 0
    const heightPct = viewport ? (dy / viewport.clientHeight) * 100 : 0
    setPan(clampPan({ x: drag.panX + widthPct, y: drag.panY + heightPct }, zoom))
  }

  function handlePointerUp(e: React.PointerEvent) {
    const drag = dragState.current
    dragState.current = null
    if (!drag || drag.moved) return // treat as a click only if the pointer barely moved

    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const xFrac = (e.clientX - rect.left) / rect.width
    const yFrac = (e.clientY - rect.top) / rect.height
    if (xFrac < 0 || xFrac > 1 || yFrac < 0 || yFrac > 1) return

    const lonIdx = Math.min(nLon - 1, Math.max(0, Math.floor(xFrac * nLon)))
    const rowFromTop = Math.min(nLat - 1, Math.max(0, Math.floor(yFrac * nLat)))
    const latIdx = nLat - 1 - rowFromTop
    if (land[latIdx * nLon + lonIdx] !== 0) return // ignore clicks on land

    onCellClick(ds.lats[latIdx], ds.lons[lonIdx])
  }

  function zoomBy(factor: number) {
    setZoom((z) => {
      const nz = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z * factor))
      setPan((p) => clampPan(p, nz))
      return nz
    })
  }

  function resetView() {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }

  const showGrid = zoom >= GRID_VISIBLE_FROM_ZOOM

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-100">
          {varDef.label}
          {depthLabel} — {ds.times[timeIdx]}
          {loading && (
            <span className="ml-3 inline-flex align-middle text-xs font-normal text-slate-200">
              <PanelLoader label="Loading" />
            </span>
          )}
          {error && <span className="ml-2 text-xs font-normal text-coral-500">failed to load: {error}</span>}
        </h3>
        <div className="flex items-center gap-1.5 text-xs text-slate-200">
          <ZoomButton onClick={() => zoomBy(1 / 1.5)} label="Zoom out">−</ZoomButton>
          <span className="w-12 text-center tabular-nums">{zoom.toFixed(1)}x</span>
          <ZoomButton onClick={() => zoomBy(1.5)} label="Zoom in">+</ZoomButton>
          <button
            onClick={resetView}
            className="ml-1 rounded-lg border border-abyss-500 px-2.5 py-1 text-xs text-slate-200 hover:border-surf-500 hover:text-surf-400"
          >
            Reset
          </button>
        </div>
      </div>

      <div
        ref={viewportRef}
        className="relative touch-none select-none overflow-hidden rounded-xl border border-abyss-600/60"
        style={{ aspectRatio: `${nLon} / ${nLat}`, cursor: dragState.current?.moved ? 'grabbing' : 'grab' }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        <div
          className="absolute inset-0"
          style={{
            transform: `translate(${pan.x}%, ${pan.y}%) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: dragState.current ? 'none' : 'transform 120ms ease-out',
          }}
        >
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            style={{ imageRendering: 'auto' }}
          />
          {showGrid && (
            <div
              className="pointer-events-none absolute inset-0"
              style={{
                backgroundImage:
                  'linear-gradient(to right, rgba(255,255,255,0.12) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.12) 1px, transparent 1px)',
                backgroundSize: `${100 / nLon}% ${100 / nLat}%`,
              }}
            />
          )}
          <div
            className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_2px_rgba(0,0,0,0.5)]"
            style={{ left: `${markerXPct}%`, top: `${markerYPct}%`, width: `${16 / zoom}px`, height: `${16 / zoom}px` }}
          />
        </div>
      </div>

      <div className="mt-3 flex items-center gap-3">
        <ColorbarLegend colormapName={varDef.colormap} min={min} max={max} unit={varDef.unit} symmetric={symmetric} />
      </div>
      <p className="mt-2 text-xs text-slate-300">
        Grey = land or below the local seafloor. White dot marks the selected point. Use +/- to zoom, drag to pan,
        click an ocean cell to open its full depth profile.
      </p>
    </div>
  )
}

function ZoomButton({ onClick, label, children }: { onClick: () => void; label: string; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="flex h-7 w-7 items-center justify-center rounded-lg border border-abyss-500 text-sm text-slate-300 hover:border-surf-500 hover:text-surf-400"
    >
      {children}
    </button>
  )
}

function ColorbarLegend({
  colormapName, min, max, unit, symmetric,
}: { colormapName: MapVariableDef['colormap']; min: number; max: number; unit: string; symmetric: boolean }) {
  const stops = Array.from({ length: 12 }, (_, i) => colormap(colormapName, i / 11))
  return (
    <div className="flex w-full items-center gap-3 text-xs text-slate-200">
      <span>{symmetric ? (-Math.abs(min)).toFixed(2) : min.toFixed(2)}</span>
      <div
        className="h-3 flex-1 rounded-full"
        style={{ background: `linear-gradient(to right, ${stops.join(', ')})` }}
      />
      <span>{max.toFixed(2)} {unit}</span>
    </div>
  )
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function parseRgb(rgb: string): [number, number, number] {
  const m = rgb.match(/\d+/g)!
  return [Number(m[0]), Number(m[1]), Number(m[2])]
}
