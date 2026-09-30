import { DEPTHS_M, DOMAIN, type Dataset, type MapVarKey } from './types'

// Synthetic dataset shaped exactly like the real one dashboard/data.py reads from Zarr, so
// swapping this generator for a `fetch()` against a real backend later is a data-source swap,
// not a component rewrite. Matches the real pipeline's 0.25deg resolution (configs/base.yaml) —
// this is a UI/UX scaffold, not a real-data view, but the grid itself should look production-scale.
const MOCK_RES_DEG = 0.25
const N_DAYS = 10
const START_DATE = new Date('2024-06-01T00:00:00Z')

function range(min: number, max: number, step: number): number[] {
  const out: number[] = []
  for (let v = min; v <= max + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000)
  return out
}

// Cheap smooth pseudo-noise (sum of a couple of low-frequency sine waves) — good enough for a
// visually plausible mock field without pulling in a real noise library. Kept low-frequency
// relative to the grid spacing so it reads as smooth eddies, not a high-frequency moire pattern.
function noise2d(x: number, y: number, seed: number): number {
  return (
    Math.sin(x * 0.35 + seed) * Math.cos(y * 0.28 + seed * 1.3) * 0.6 +
    Math.sin(x * 0.6 - seed * 0.5) * Math.cos(y * 0.5 + seed) * 0.4
  )
}

/** Rough, stylized "is this cell land" for the Indian subcontinent + Arabian peninsula, purely
 * for a plausible-looking mock map — not a real coastline. */
function isLandMock(lat: number, lon: number): boolean {
  const coastNoise = noise2d(lat * 0.5, lon * 0.5, 7) * 3
  // Indian subcontinent: wide in the north, tapering to a point near the south.
  const peninsulaWidth = Math.max(0, 22 - lat) * 1.3
  const peninsulaCenter = 78
  if (lat > 8 && lat < 30 && Math.abs(lon - peninsulaCenter + coastNoise) < 6 + peninsulaWidth * 0.35) return true
  // Sri Lanka.
  if (lat > 5.8 && lat < 9.8 && Math.abs(lon - 80.7) < 1.3) return true
  // Arabian peninsula (west edge of domain).
  if (lon < 60 && lat > 12 && lat < 30 && lon - 45 < 12 + coastNoise) return true
  // Myanmar / SE Asia (east edge of domain).
  if (lon > 92 && lat > 10 && lon - 92 + coastNoise * 1.5 < (lat - 5) * 0.6) return true
  return false
}

function thermoclineProfile(depth: number, surfaceTemp: number): number {
  // Warm mixed layer, sharp drop through the thermocline (~50-300m), cold deep water.
  if (depth <= 30) return surfaceTemp
  const deepTemp = 3.5
  const midDrop = 1 / (1 + Math.exp(-(depth - 150) / 60))
  return surfaceTemp - (surfaceTemp - deepTemp) * midDrop
}

export function generateMockDataset(): Dataset {
  const lats = Float32Array.from(range(DOMAIN.latMin, DOMAIN.latMax, MOCK_RES_DEG))
  const lons = Float32Array.from(range(DOMAIN.lonMin, DOMAIN.lonMax, MOCK_RES_DEG))
  const nLat = lats.length
  const nLon = lons.length
  const nCell = nLat * nLon
  const depths = [...DEPTHS_M]
  const nDepth = depths.length

  const times: string[] = []
  for (let d = 0; d < N_DAYS; d++) {
    const t = new Date(START_DATE)
    t.setUTCDate(t.getUTCDate() + d)
    times.push(t.toISOString().slice(0, 10))
  }

  const qualityFlag: Uint8Array[] = []
  const landAt = new Uint8Array(nCell)
  for (let i = 0; i < nLat; i++) {
    for (let j = 0; j < nLon; j++) {
      landAt[i * nLon + j] = isLandMock(lats[i], lons[j]) ? 1 : 0
    }
  }
  for (let t = 0; t < N_DAYS; t++) qualityFlag.push(landAt.slice())

  const fields: Record<MapVarKey, Float32Array[]> = {
    temp: [], temp_std: [], error: [], glorys: [], mld: [], d20: [], tchp: [],
  }

  for (let t = 0; t < N_DAYS; t++) {
    const tempArr = new Float32Array(nCell * nDepth)
    const stdArr = new Float32Array(nCell * nDepth)
    const glorysArr = new Float32Array(nCell * nDepth)
    const mldArr = new Float32Array(nCell)
    const d20Arr = new Float32Array(nCell)
    const tchpArr = new Float32Array(nCell)

    for (let i = 0; i < nLat; i++) {
      for (let j = 0; j < nLon; j++) {
        const cell = i * nLon + j
        if (landAt[cell]) continue
        const lat = lats[i]
        const lon = lons[j]
        const seasonalDay = t / N_DAYS
        const eddy = noise2d(lat * 1.5, lon * 1.5, seasonalDay * 6)
        // Warmer near the equator, a mesoscale eddy pattern, mild day-to-day drift.
        const surfaceTemp = 27.5 - Math.abs(lat - 10) * 0.28 + eddy * 1.8 + Math.sin(seasonalDay * Math.PI * 2) * 0.4

        for (let k = 0; k < nDepth; k++) {
          const depth = depths[k]
          const idx = (i * nLon + j) * nDepth + k
          const reconstruction = thermoclineProfile(depth, surfaceTemp)
          const bias = noise2d(lat * 0.8 + 3, lon * 0.8 + 3, k) * 0.15
          const uncertainty = 0.15 + (0.35 * Math.exp(-Math.pow((depth - 120) / 90, 2))) + depth * 0.0004
          tempArr[idx] = reconstruction
          glorysArr[idx] = reconstruction + bias
          stdArr[idx] = uncertainty
        }

        mldArr[cell] = 25 + Math.max(0, eddy) * 30 + Math.abs(lat - 15) * 0.8
        d20Arr[cell] = 60 + Math.max(0, eddy) * 45 + Math.abs(lat - 15) * 1.2
        tchpArr[cell] = Math.max(0, 40 + eddy * 35 - Math.abs(lat - 15) * 1.5)
      }
    }

    fields.temp.push(tempArr)
    fields.temp_std.push(stdArr)
    fields.glorys.push(glorysArr)
    fields.error.push(new Float32Array(0)) // derived on the fly (temp - glorys), see mapField()
    fields.mld.push(mldArr)
    fields.d20.push(d20Arr)
    fields.tchp.push(tchpArr)
  }

  return {
    times,
    depths,
    lats,
    lons,
    fields,
    qualityFlag,
    attrs: {
      dataSource: 'synthetic',
      modelVersion: 'ui-mock-0.1',
      syntheticWarning:
        'This view uses synthetic placeholder data for UI/UX design (Chunks 4-6 have not produced real model output yet)',
    },
  }
}
