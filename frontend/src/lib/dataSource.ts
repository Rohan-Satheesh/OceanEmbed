import { generateMockDataset } from './mockData'
import * as sel from './selectors'
import type { ProfilePoint, RegionalPoint, SeriesPoint } from './selectors'
import { REGIONS, type Attrs, type MapVarKey, type Meta } from './types'

/** Where the UI gets its numbers. Two implementations: the real backend (`api/server.py`) and an
 * in-browser synthetic generator used when the backend isn't reachable. Same contract, so components
 * never know which one they're talking to. */
export interface DataSource {
  kind: 'api' | 'mock'
  meta: Meta
  /** (lat, lon) grid, row-major, NaN where there is no data. */
  mapField(varKey: MapVarKey, timeIdx: number, depthIdx: number): Promise<Float32Array>
  profile(timeIdx: number, lat: number, lon: number): Promise<ProfilePoint[]>
  timeseries(depthIdx: number, lat: number, lon: number): Promise<SeriesPoint[]>
  regional(region: keyof typeof REGIONS, depthIdx: number): Promise<RegionalPoint[]>
}

const num = (v: number | null): number => (v === null ? NaN : v)

// ---- mock ---------------------------------------------------------------------------------------

export function createMockSource(): DataSource {
  const ds = generateMockDataset()
  const meta: Meta = {
    times: ds.times,
    depths: ds.depths,
    lats: ds.lats,
    lons: ds.lons,
    land: ds.qualityFlag[0],
    attrs: ds.attrs,
  }
  return {
    kind: 'mock',
    meta,
    mapField: async (v, t, d) => sel.mapField(ds, v, t, d),
    profile: async (t, lat, lon) => sel.profile(ds, t, lat, lon),
    timeseries: async (d, lat, lon) => sel.timeseries(ds, d, lat, lon),
    regional: async (region, d) => sel.regionalSummary(ds, region, d),
  }
}

// ---- real backend -------------------------------------------------------------------------------

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`)
  return (await res.json()) as T
}

async function getBytes(path: string): Promise<ArrayBuffer> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`)
  return res.arrayBuffer()
}

export async function createApiSource(): Promise<DataSource> {
  const [m, landBuf] = await Promise.all([
    getJson<{ times: string[]; depths: number[]; lats: number[]; lons: number[]; attrs: Attrs }>('/meta'),
    getBytes('/land'),
  ])
  const meta: Meta = {
    times: m.times,
    depths: m.depths,
    lats: Float32Array.from(m.lats),
    lons: Float32Array.from(m.lons),
    land: new Uint8Array(landBuf),
    attrs: m.attrs,
  }

  // Fields are ~100 kB each; keep the ones already fetched so scrubbing back and forth is instant.
  const fieldCache = new Map<string, Float32Array>()

  return {
    kind: 'api',
    meta,
    async mapField(v, t, d) {
      const depthFree = v === 'mld' || v === 'd20' || v === 'tchp'
      const key = `${v}|${t}|${depthFree ? 0 : d}`
      const hit = fieldCache.get(key)
      if (hit) return hit
      const out = new Float32Array(await getBytes(`/field?var=${v}&time=${t}&depth=${depthFree ? 0 : d}`))
      if (fieldCache.size > 120) fieldCache.delete(fieldCache.keys().next().value as string)
      fieldCache.set(key, out)
      return out
    },
    async profile(t, lat, lon) {
      const r = await getJson<{ depthM: number[]; temp: (number | null)[]; std: (number | null)[]; glorys: (number | null)[] }>(
        `/profile?time=${t}&lat=${lat}&lon=${lon}`,
      )
      return r.depthM.map((depthM, k) => ({ depthM, temp: num(r.temp[k]), std: num(r.std[k]), glorys: num(r.glorys[k]) }))
    },
    async timeseries(d, lat, lon) {
      const r = await getJson<{ time: string[]; temp: (number | null)[]; std: (number | null)[]; glorys: (number | null)[] }>(
        `/timeseries?depth=${d}&lat=${lat}&lon=${lon}`,
      )
      return r.time.map((time, i) => ({ time, temp: num(r.temp[i]), std: num(r.std[i]), glorys: num(r.glorys[i]) }))
    },
    async regional(region, d) {
      const r = await getJson<{ time: string[]; reconstruction: (number | null)[]; glorys: (number | null)[]; rmse: (number | null)[] }>(
        `/regional?region=${String(region)}&depth=${d}`,
      )
      return r.time.map((time, i) => ({
        time,
        reconstruction: num(r.reconstruction[i]),
        glorys: num(r.glorys[i]),
        rmse: num(r.rmse[i]),
      }))
    },
  }
}

/** Real backend if it answers, otherwise the synthetic generator (so `npm run dev` alone still works). */
export async function connectDataSource(timeoutMs = 8000): Promise<DataSource> {
  try {
    const timeout = new Promise<never>((_, reject) => setTimeout(() => reject(new Error('timed out')), timeoutMs))
    return await Promise.race([createApiSource(), timeout])
  } catch (err) {
    console.warn('OceanEmbed API unreachable, falling back to synthetic mock data:', err)
    return createMockSource()
  }
}
