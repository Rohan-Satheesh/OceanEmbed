import type { ColormapName } from './types'

type RGB = [number, number, number]

// Control-point approximations of the matplotlib colormaps used by the Streamlit dashboard
// (dashboard/data.py MAP_VARIABLES), sampled at ~8-10 stops each. Close enough visually for a
// UI heatmap; not a scientific-precision reproduction of the originals.
const STOPS: Record<ColormapName, RGB[]> = {
  turbo: [
    [48, 18, 59], [70, 107, 227], [39, 174, 219], [42, 209, 156],
    [140, 222, 66], [222, 200, 45], [239, 120, 39], [204, 38, 24], [122, 4, 3],
  ],
  magma: [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191],
  ],
  viridis: [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37],
  ],
  ylorrd: [
    [255, 255, 178], [254, 217, 118], [254, 178, 76],
    [253, 141, 60], [252, 78, 42], [227, 26, 28], [177, 0, 38],
  ],
  // Diverging: negative = blue, zero = white, positive = red. Caller passes t already
  // normalized to [0,1] around a symmetric range for the `error` variable.
  rdbu: [
    [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240],
    [247, 247, 247],
    [253, 219, 199], [244, 165, 130], [214, 96, 77], [178, 24, 43], [103, 0, 31],
  ],
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

/** t in [0, 1] -> "rgb(r, g, b)" string. */
export function colormap(name: ColormapName, t: number): string {
  const stops = STOPS[name]
  const clamped = Math.min(1, Math.max(0, t))
  const scaled = clamped * (stops.length - 1)
  const i0 = Math.floor(scaled)
  const i1 = Math.min(stops.length - 1, i0 + 1)
  const frac = scaled - i0
  const [r0, g0, b0] = stops[i0]
  const [r1, g1, b1] = stops[i1]
  const r = Math.round(lerp(r0, r1, frac))
  const g = Math.round(lerp(g0, g1, frac))
  const b = Math.round(lerp(b0, b1, frac))
  return `rgb(${r}, ${g}, ${b})`
}

export const LAND_COLOR = '#2a3244'
