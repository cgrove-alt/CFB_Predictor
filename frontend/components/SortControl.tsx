'use client'

export type SortOption = 'edge' | 'confidence' | 'time' | 'default'

interface SortControlProps {
  value: SortOption
  onChange: (value: SortOption) => void
}

const SORT_OPTIONS: { value: SortOption; label: string; icon: React.ReactNode }[] = [
  {
    value: 'edge',
    label: 'Highest Edge',
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
  },
  {
    value: 'confidence',
    label: 'Highest Confidence',
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
  },
  {
    value: 'time',
    label: 'Start Time',
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
]

export function SortControl({ value, onChange }: SortControlProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-slate-400 hidden sm:inline">Sort by:</span>
      <div className="flex rounded-lg bg-slate-800 p-1 gap-1">
        {SORT_OPTIONS.map((option) => (
          <button
            key={option.value}
            onClick={() => onChange(option.value)}
            className={`flex items-center gap-1.5 px-3 py-2 sm:py-1.5 rounded-md text-sm font-medium transition-all ${
              value === option.value
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-700'
            }`}
            title={option.label}
          >
            {option.icon}
            <span className="hidden md:inline">{option.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/**
 * Compact dropdown version for mobile
 */
interface SortDropdownProps {
  value: SortOption
  onChange: (value: SortOption) => void
}

export function SortDropdown({ value, onChange }: SortDropdownProps) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-slate-400">Sort:</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as SortOption)}
        className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-white text-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
      >
        <option value="edge">Highest Edge</option>
        <option value="confidence">Highest Confidence</option>
        <option value="time">Start Time</option>
        <option value="default">Default</option>
      </select>
    </div>
  )
}
