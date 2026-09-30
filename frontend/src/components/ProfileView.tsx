import { ErrorPanel, LoadingPanel } from './AsyncPanels'
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DataSource } from '../lib/dataSource'
import { useAsync } from '../lib/useAsync'

interface ProfileViewProps {
  src: DataSource
  timeIdx: number
  lat: number
  lon: number
  onLand: boolean
}

export function ProfileView({ src, timeIdx, lat, lon, onLand }: ProfileViewProps) {
  const ds = src.meta
  const { data: points, error } = useAsync(
    () => (onLand ? Promise.resolve([]) : src.profile(timeIdx, lat, lon)),
    [src, timeIdx, lat, lon, onLand],
  )

  if (onLand) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-sm text-slate-200">
        No data at this point (land).
      </div>
    )
  }

  if (error) return <ErrorPanel message={error} />
  if (!points) return <LoadingPanel />

  const rows = points.map((p) => ({
    ...p,
    band: [p.temp - 1.96 * p.std, p.temp + 1.96 * p.std] as [number, number],
  }))

  return (
    <div className="glass-panel rounded-2xl p-5">
      <h3 className="mb-1 text-sm font-semibold text-slate-100">
        Vertical profile — {lat.toFixed(2)}N {lon.toFixed(2)}E — {ds.times[timeIdx]}
      </h3>
      <p className="mb-4 text-xs text-slate-300">Shaded band = +/-1.96 std (MC dropout). Dashed = GLORYS reference.</p>
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={rows} layout="vertical" margin={{ left: 8, right: 24, top: 8, bottom: 8 }}>
          <CartesianGrid stroke="#1b4657" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="temp"
            domain={['auto', 'auto']}
            stroke="#e2e8f0"
            fontSize={12}
            label={{ value: 'temperature (deg C)', position: 'insideBottom', offset: -4, fill: '#cbd5e1', fontSize: 11 }}
          />
          <YAxis type="category" dataKey="depthM" stroke="#e2e8f0" fontSize={12} width={56} label={{ value: 'depth (m)', angle: -90, position: 'insideLeft', fill: '#cbd5e1', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#0c2836', border: '1px solid #1b4657', borderRadius: 10, fontSize: 12 }}
            labelStyle={{ color: '#e2e8f0' }}
            formatter={(value, name) => {
              if (name === 'band' && Array.isArray(value)) return [`${value[0].toFixed(2)} - ${value[1].toFixed(2)}`, 'uncertainty band']
              return [Number(value).toFixed(2), name]
            }}
          />
          <Area dataKey="band" stroke="none" fill="#38e3d8" fillOpacity={0.15} />
          <Line dataKey="glorys" stroke="#ff7a59" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 2 }} name="GLORYS" />
          <Line dataKey="temp" stroke="#38e3d8" strokeWidth={2.5} dot={{ r: 3 }} name="reconstruction" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
