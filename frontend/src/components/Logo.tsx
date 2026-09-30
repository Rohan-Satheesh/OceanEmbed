// OceanEmbed mark: two stacked waves under a sun disc, on a deep-sea tile.
export function Logo({ size = 40 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" role="img" aria-label="OceanEmbed" className="shrink-0">
      <rect width="48" height="48" rx="12" fill="#123444" />
      <circle cx="31" cy="16" r="6" fill="#f3e4c0" />
      <path d="M6 28 Q12 22 18 28 T30 28 T42 28" fill="none" stroke="#38e3d8" strokeWidth="3" strokeLinecap="round" />
      <path d="M6 37 Q12 31 18 37 T30 37 T42 37" fill="none" stroke="#f3e4c0" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}
