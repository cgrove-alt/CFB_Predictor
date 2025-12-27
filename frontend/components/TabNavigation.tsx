'use client'

export type TabType = 'spreads' | 'totals' | 'results'

interface TabNavigationProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

export function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  const tabs: { id: TabType; label: string; icon: string }[] = [
    { id: 'spreads', label: 'Spread Picks', icon: '🏈' },
    { id: 'totals', label: 'Totals (O/U)', icon: '📊' },
    { id: 'results', label: 'Results', icon: '📜' },
  ]

  return (
    <div className="flex gap-1 bg-slate-800 rounded-lg p-1">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`flex-1 px-3 sm:px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
            activeTab === tab.id
              ? 'bg-slate-700 text-white shadow-sm'
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <span className="sm:mr-2">{tab.icon}</span>
          <span className="hidden sm:inline">{tab.label}</span>
        </button>
      ))}
    </div>
  )
}
