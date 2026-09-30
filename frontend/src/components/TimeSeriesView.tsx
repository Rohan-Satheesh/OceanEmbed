import { ErrorPanel, LoadingPanel } from './AsyncPanels'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DataSource } from '../lib/dataSource'
import { useAsync } from '../lib/useAsync'

interface TimeSeriesViewProps {
  src: DataSource
  depthIdx: number
  lat: number
  lon: number
  onLand: boolean
}

export function TimeSeriesView({ src, depthIdx, lat, lon, onLand }: TimeSeriesViewProps) {
  const ds = src.meta
  const { data: rows, error } = useAsync(
    () => (onLand ? Promise.resolve([]) : src.timeseries(depthIdx, lat, lon)),
    [src, depthIdx, lat, lon, onLand],
  )

  if (onLand) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-sm text-slate-200">
        No data at this point (land).
      </div>
    )
  }

  if (error) return <ErrorPanel message={error} />
  if (!rows) return <LoadingPanel />

  return (
    <div className="glass-panel rounded-2xl p-5">
      <h3 className="mb-1 text-sm font-semibold text-slate-100">
        Time series — {lat.toFixed(2)}N {lon.toFixed(2)}E at {ds.depths[depthIdx]} m
      </h3>
      <p className="mb-4 text-xs text-slate-300">Reconstruction vs GLORYS at the selected point and depth (evaluation days only).</p>
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={rows} margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
          <CartesianGrid stroke="#1b4657" strokeDasharray="3 3" />
          <XAxis dataKey="time" stroke="#e2e8f0" fontSize={11} />
          <YAxis
            stroke="#e2e8f0"
            fontSize={12}
            domain={['auto', 'auto']}
            label={{ value: `temp (deg C) @ ${ds.depths[depthIdx]}m`, angle: -90, position: 'insideLeft', fill: '#cbd5e1', fontSize: 11 }}
          />
          <Tooltip contentStyle={{ background: '#0c2836', border: '1px solid #1b4657', borderRadius: 10, fontSize: 12 }} labelStyle={{ color: '#e2e8f0' }} />
          <Legend wrapperStyle={{ fontSize: 12, color: '#e2e8f0' }} />
          <Line type="monotone" dataKey="temp" name="reconstruction" stroke="#38e3d8" strokeWidth={2.5} dot={false} />
          <Line type="monotone" dataKey="glorys" name="GLORYS" stroke="#ff7a59" strokeWidth={2} strokeDasharray="5 4" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
