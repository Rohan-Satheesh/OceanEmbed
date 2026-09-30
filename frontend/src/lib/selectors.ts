import { REGIONS, type Dataset, type MapVarKey, type Meta } from './types'

function nearestIndex(values: ArrayLike<number>, target: number): number {
  let best = 0
  let bestDist = Infinity
  for (let i = 0; i < values.length; i++) {
    const dist = Math.abs(values[i] - target)
    if (dist < bestDist) {
      bestDist = dist
      best = i
    }
  }
  return best
}

function nearestCell(ds: Pick<Dataset, 'lats' | 'lons'>, lat: number, lon: number): { latIdx: number; lonIdx: number; lat: number; lon: number } {
  const latIdx = nearestIndex(ds.lats, lat)
  const lonIdx = nearestIndex(ds.lons, lon)
  return { latIdx, lonIdx, lat: ds.lats[latIdx], lon: ds.lons[lonIdx] }
}

export function isOcean(meta: Meta, lat: number, lon: number): boolean {
  const { latIdx, lonIdx } = nearestCell(meta, lat, lon)
  return meta.land[latIdx * meta.lons.length + lonIdx] === 0
}

/** 2D (lat, lon) grid for the map view. `error` is derived on the fly (temp - glorys). */
export function mapField(ds: Dataset, varKey: MapVarKey, timeIdx: number, depthIdx: number): Float32Array {
  const nLat = ds.lats.length
  const nLon = ds.lons.length
  const nDepth = ds.depths.length
  const depthFree = varKey === 'mld' || varKey === 'd20' || varKey === 'tchp'

  if (depthFree) return ds.fields[varKey][timeIdx]

  if (varKey === 'error') {
    const temp = ds.fields.temp[timeIdx]
    const glorys = ds.fields.glorys[timeIdx]
    const out = new Float32Array(nLat * nLon)
    for (let cell = 0; cell < nLat * nLon; cell++) {
      out[cell] = temp[cell * nDepth + depthIdx] - glorys[cell * nDepth + depthIdx]
    }
    return out
  }

  const src = ds.fields[varKey][timeIdx]
  const out = new Float32Array(nLat * nLon)
  for (let cell = 0; cell < nLat * nLon; cell++) out[cell] = src[cell * nDepth + depthIdx]
  return out
}

export interface ProfilePoint {
  depthM: number
  temp: number
  std: number
  glorys: number
}

export function profile(ds: Dataset, timeIdx: number, lat: number, lon: number): ProfilePoint[] {
  const { latIdx, lonIdx } = nearestCell(ds, lat, lon)
  const nLon = ds.lons.length
  const nDepth = ds.depths.length
  const cell = latIdx * nLon + lonIdx
  const temp = ds.fields.temp[timeIdx]
  const std = ds.fields.temp_std[timeIdx]
  const glorys = ds.fields.glorys[timeIdx]
  return ds.depths.map((depthM, k) => ({
    depthM,
    temp: temp[cell * nDepth + k],
    std: std[cell * nDepth + k],
    glorys: glorys[cell * nDepth + k],
  }))
}

export interface SeriesPoint {
  time: string
  temp: number
  std: number
  glorys: number
}

export function timeseries(ds: Dataset, depthIdx: number, lat: number, lon: number): SeriesPoint[] {
  const { latIdx, lonIdx } = nearestCell(ds, lat, lon)
  const nLon = ds.lons.length
  const nDepth = ds.depths.length
  const cell = latIdx * nLon + lonIdx
  return ds.times.map((time, t) => ({
    time,
    temp: ds.fields.temp[t][cell * nDepth + depthIdx],
    std: ds.fields.temp_std[t][cell * nDepth + depthIdx],
    glorys: ds.fields.glorys[t][cell * nDepth + depthIdx],
  }))
}

export interface RegionalPoint {
  time: string
  reconstruction: number
  glorys: number
  rmse: number
}

export function regionalSummary(ds: Dataset, region: keyof typeof REGIONS, depthIdx: number): RegionalPoint[] {
  const { latMin, latMax, lonMin, lonMax } = REGIONS[region]
  const nLon = ds.lons.length
  const nDepth = ds.depths.length
  const cells: number[] = []
  for (let i = 0; i < ds.lats.length; i++) {
    if (ds.lats[i] < latMin || ds.lats[i] > latMax) continue
    for (let j = 0; j < nLon; j++) {
      if (ds.lons[j] < lonMin || ds.lons[j] > lonMax) continue
      cells.push(i * nLon + j)
    }
  }

  return ds.times.map((time, t) => {
    let sumRecon = 0
    let sumGlorys = 0
    let sumSqErr = 0
    let n = 0
    const land = ds.qualityFlag[t]
    const temp = ds.fields.temp[t]
    const glorys = ds.fields.glorys[t]
    for (const cell of cells) {
      if (land[cell] !== 0) continue
      const r = temp[cell * nDepth + depthIdx]
      const g = glorys[cell * nDepth + depthIdx]
      sumRecon += r
      sumGlorys += g
      sumSqErr += (r - g) ** 2
      n++
    }
    return {
      time,
      reconstruction: n ? sumRecon / n : NaN,
      glorys: n ? sumGlorys / n : NaN,
      rmse: n ? Math.sqrt(sumSqErr / n) : NaN,
    }
  })
}

export interface Patch {
  latStart: number
  lonStart: number
  nLat: number
  nLon: number
  lats: number[]
  lons: number[]
  /** Surface (depth index 0) temperature at (patch row, patch col). */
  surfaceTemp: (row: number, col: number) => number
  /** Full-depth temperature at (patch row, patch col, depth index). */
  temp: (row: number, col: number, depthIdx: number) => number
  isLand: (row: number, col: number) => boolean
}

/** A square-ish window of the grid around (lat, lon), radius cells in each direction, clamped to
 * the domain edges — used by the 3D block view so it doesn't have to hold/index the whole grid. */
export function patchAround(ds: Dataset, timeIdx: number, lat: number, lon: number, radius: number): Patch {
  const { latIdx: centerLatIdx, lonIdx: centerLonIdx } = nearestCell(ds, lat, lon)
  const nLonFull = ds.lons.length
  const nDepth = ds.depths.length

  const latStart = Math.max(0, centerLatIdx - radius)
  const latEnd = Math.min(ds.lats.length - 1, centerLatIdx + radius)
  const lonStart = Math.max(0, centerLonIdx - radius)
  const lonEnd = Math.min(nLonFull - 1, centerLonIdx + radius)
  const nLat = latEnd - latStart + 1
  const nLon = lonEnd - lonStart + 1

  const temp = ds.fields.temp[timeIdx]
  const land = ds.qualityFlag[timeIdx]

  return {
    latStart,
    lonStart,
    nLat,
    nLon,
    lats: Array.from({ length: nLat }, (_, r) => ds.lats[latStart + r]),
    lons: Array.from({ length: nLon }, (_, c) => ds.lons[lonStart + c]),
    surfaceTemp: (row, col) => {
      const cell = (latStart + row) * nLonFull + (lonStart + col)
      return temp[cell * nDepth + 0]
    },
    temp: (row, col, depthIdx) => {
      const cell = (latStart + row) * nLonFull + (lonStart + col)
      return temp[cell * nDepth + depthIdx]
    },
    isLand: (row, col) => land[(latStart + row) * nLonFull + (lonStart + col)] !== 0,
  }
}

export function defaultLocation(ds: Meta): { lat: number; lon: number } {
  const candidates: [number, number][] = [[15.0, 88.0], [15.0, 65.0]]
  for (const [lat, lon] of candidates) {
    if (isOcean(ds, lat, lon)) return { lat, lon }
  }
  const land = ds.land
  const nLon = ds.lons.length
  for (let i = 0; i < ds.lats.length; i++) {
    for (let j = 0; j < nLon; j++) {
      if (land[i * nLon + j] === 0) return { lat: ds.lats[i], lon: ds.lons[j] }
    }
  }
  return { lat: ds.lats[0], lon: ds.lons[0] }
}
