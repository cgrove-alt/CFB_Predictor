'use client'

interface WeekSelectorProps {
  season: number
  week: number
  seasonType: string
  onSeasonChange: (season: number) => void
  onWeekChange: (week: number) => void
  onSeasonTypeChange: (type: string) => void
}

export function WeekSelector({
  season,
  week,
  seasonType,
  onSeasonChange,
  onWeekChange,
  onSeasonTypeChange,
}: WeekSelectorProps) {
  const currentYear = new Date().getFullYear()
  const seasons = [currentYear, currentYear - 1, currentYear - 2]
  const regularWeeks = Array.from({ length: 15 }, (_, i) => i + 1)
  const postseasonWeeks = [1] // Bowl season

  const weeks = seasonType === 'postseason' ? postseasonWeeks : regularWeeks

  return (
    <div className="flex flex-wrap gap-4 items-center bg-slate-800 rounded-lg p-4">
      {/* Season */}
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-400">Season:</label>
        <select
          value={season}
          onChange={(e) => onSeasonChange(Number(e.target.value))}
          className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
        >
          {seasons.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      {/* Season Type */}
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-400">Type:</label>
        <div className="flex rounded-lg overflow-hidden">
          <button
            onClick={() => onSeasonTypeChange('regular')}
            className={`px-3 py-1.5 text-sm transition-colors ${
              seasonType === 'regular'
                ? 'bg-emerald-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Regular
          </button>
          <button
            onClick={() => onSeasonTypeChange('postseason')}
            className={`px-3 py-1.5 text-sm transition-colors ${
              seasonType === 'postseason'
                ? 'bg-emerald-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Postseason
          </button>
        </div>
      </div>

      {/* Week */}
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-400">Week:</label>
        <select
          value={week}
          onChange={(e) => onWeekChange(Number(e.target.value))}
          className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
        >
          {weeks.map((w) => (
            <option key={w} value={w}>
              {seasonType === 'postseason' ? 'Bowl Games' : `Week ${w}`}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
