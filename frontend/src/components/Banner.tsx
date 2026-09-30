interface BannerProps {
  warning?: string
}

export function Banner({ warning }: BannerProps) {
  if (!warning) return null
  return (
    <div className="flex items-start gap-3 rounded-xl border border-coral-600/40 bg-coral-600/10 px-4 py-3 text-sm text-coral-500">
      <span className="mt-0.5 text-base leading-none">&#9888;</span>
      <p>
        <strong className="font-semibold">{warning}.</strong> Do not read these maps or scores as real ocean results.
      </p>
    </div>
  )
}
