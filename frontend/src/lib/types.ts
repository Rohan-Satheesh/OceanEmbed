// Mirrors the xarray Dataset shape written by `python -m evaluation.predict`
// (see dashboard/data.py) — keep in sync so swapping mock data for a real
// backend later is a data-source change, not a component rewrite.

export type MapVarKey = 'temp' | 'temp_std' | 'error' | 'glorys' | 'mld' | 'd20' | 'tchp'

export interface MapVariableDef {
  key: MapVarKey
  label: string
  colormap: ColormapName
  unit: string
  depthFree: boolean
}

export const MAP_VARIABLES: MapVariableDef[] = [
  { key: 'temp', label: 'Temperature', colormap: 'turbo', unit: 'deg C', depthFree: false },
  { key: 'temp_std', label: 'Uncertainty (MC-dropout std)', colormap: 'magma', unit: 'deg C', depthFree: false },
  { key: 'error', label: 'Difference vs GLORYS (reconstruction - GLORYS)', colormap: 'rdbu', unit: 'deg C', depthFree: false },
  { key: 'glorys', label: 'GLORYS (reference)', colormap: 'turbo', unit: 'deg C', depthFree: false },
  { key: 'mld', label: 'Mixed layer depth', colormap: 'viridis', unit: 'm', depthFree: true },
  { key: 'd20', label: '20C isotherm depth (D20)', colormap: 'viridis', unit: 'm', depthFree: true },
  { key: 'tchp', label: 'Tropical cyclone heat potential', colormap: 'ylorrd', unit: 'kJ/cm2', depthFree: true },
]

export type ColormapName = 'turbo' | 'magma' | 'rdbu' | 'viridis' | 'ylorrd'

// PRD Section 4.4 — 15 standard reconstruction depths.
export const DEPTHS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] as const

// configs/base.yaml domain bounds.
export const DOMAIN = { latMin: 5.0, latMax: 30.0, lonMin: 45.0, lonMax: 105.0, resDeg: 0.25 } as const

// evaluation/metrics.py REGIONS — (lat_min, lat_max, lon_min, lon_max)
export const REGIONS: Record<string, { label: string; latMin: number; latMax: number; lonMin: number; lonMax: number }> = {
  bay_of_bengal: { label: 'Bay of Bengal', latMin: 5.0, latMax: 23.0, lonMin: 80.0, lonMax: 100.0 },
  arabian_sea: { label: 'Arabian Sea', latMin: 5.0, latMax: 25.0, lonMin: 45.0, lonMax: 77.0 },
}

export interface Dataset {
  times: string[] // ISO date strings
  depths: number[]
  lats: Float32Array
  lons: Float32Array
  // field[varKey][timeIdx] -> Float32Array of length lats.length * lons.length, row-major (lat, lon).
  // depth-resolved vars are field[varKey][timeIdx * depths.length + depthIdx].
  fields: Record<MapVarKey, Float32Array[]>
  qualityFlag: Uint8Array[] // per time, 0 = ocean/valid at surface, 1 = land/masked
  attrs: Attrs
}

export interface Attrs {
  dataSource: 'synthetic' | 'real'
  modelVersion: string
  syntheticWarning?: string | null
  note?: string | null
}

/** Everything the UI needs synchronously: grid axes, the static surface land mask, and attributes.
 * The heavy fields are fetched on demand through a `DataSource` (see dataSource.ts). */
export interface Meta {
  times: string[]
  depths: number[]
  lats: Float32Array
  lons: Float32Array
  land: Uint8Array // row-major (lat, lon); 0 = ocean at the surface, non-zero = land / masked
  attrs: Attrs
}
