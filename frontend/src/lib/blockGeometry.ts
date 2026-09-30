import * as THREE from 'three'

/**
 * Builds a quad-grid BufferGeometry (nRows x nCols vertices, triangulated, with per-vertex
 * color) from position/color callback functions. Shared by the 3D block's top "terrain" surface
 * and its four vertical cross-section walls — all five are the same shape (a displaced, colored
 * grid), just oriented and fed differently.
 */
export function buildGridMesh(
  nRows: number,
  nCols: number,
  posFn: (row: number, col: number) => [number, number, number],
  colorFn: (row: number, col: number) => [number, number, number],
): THREE.BufferGeometry {
  const positions = new Float32Array(nRows * nCols * 3)
  const colors = new Float32Array(nRows * nCols * 3)

  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < nCols; c++) {
      const idx = r * nCols + c
      const [x, y, z] = posFn(r, c)
      positions[idx * 3] = x
      positions[idx * 3 + 1] = y
      positions[idx * 3 + 2] = z
      const [cr, cg, cb] = colorFn(r, c)
      colors[idx * 3] = cr
      colors[idx * 3 + 1] = cg
      colors[idx * 3 + 2] = cb
    }
  }

  const indices: number[] = []
  for (let r = 0; r < nRows - 1; r++) {
    for (let c = 0; c < nCols - 1; c++) {
      const a = r * nCols + c
      const b = r * nCols + c + 1
      const d = (r + 1) * nCols + c
      const e = (r + 1) * nCols + c + 1
      indices.push(a, d, b, b, d, e)
    }
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geo.setIndex(indices)
  geo.computeVertexNormals()
  return geo
}

/** [r, g, b] each in 0-1, parsed from colormap()'s "rgb(r, g, b)" string output. */
export function rgbStringToUnit(rgb: string): [number, number, number] {
  const m = rgb.match(/\d+/g)!
  return [Number(m[0]) / 255, Number(m[1]) / 255, Number(m[2]) / 255]
}
