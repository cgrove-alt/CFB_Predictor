'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Header } from '@/components/Header'
import { WeekSelector } from '@/components/WeekSelector'
import { GameCard } from '@/components/GameCard'
import { TabNavigation, TabType } from '@/components/TabNavigation'
import { HeroSection } from '@/components/HeroSection'
import { apiClient } from '@/lib/api'
import { PredictionsResponse, Prediction, ResultsResponse } from '@/lib/types'
import { ResultsTable } from '@/components/ResultsTable'
import { ResultsSummary } from '@/components/ResultsSummary'

type FilterType = 'all' | 'bets' | 'leans' | 'actionable'

export default function DashboardPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Tab state
  const [activeTab, setActiveTab] = useState<TabType>('spreads')

  // Selection state
  const currentYear = new Date().getFullYear()
  const [season, setSeason] = useState(currentYear)
  const [week, setWeek] = useState(1)
  const [seasonType, setSeasonType] = useState('regular')
  const [bankroll, setBankroll] = useState(1000)

  // Data state
  const [predictions, setPredictions] = useState<PredictionsResponse | null>(null)
  const [results, setResults] = useState<ResultsResponse | null>(null)
  const [resultsLoading, setResultsLoading] = useState(false)

  // Filter state
  const [filter, setFilter] = useState<FilterType>('all')
  const [sortBy, setSortBy] = useState<'edge' | 'probability' | 'default'>('default')

  // Check authentication
  useEffect(() => {
    const isAuth = localStorage.getItem('authenticated')
    if (isAuth !== 'true') {
      router.push('/')
      return
    }

    // Detect current week (rough estimate based on date)
    const now = new Date()
    const seasonStart = new Date(currentYear, 7, 24) // Late August
    if (now >= seasonStart) {
      const weeksSinceStart = Math.floor((now.getTime() - seasonStart.getTime()) / (7 * 24 * 60 * 60 * 1000))
      const currentWeek = Math.min(Math.max(weeksSinceStart + 1, 1), 15)
      setWeek(currentWeek)

      // Check if we're in bowl season
      if (now.getMonth() === 11 && now.getDate() >= 15) {
        setSeasonType('postseason')
        setWeek(1)
      }
    }
  }, [router, currentYear])

  // Fetch predictions when selection changes
  useEffect(() => {
    const fetchPredictions = async () => {
      setIsLoading(true)
      setError(null)

      try {
        const data = await apiClient.getPredictions(season, week, seasonType, bankroll)
        setPredictions(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch predictions')
      } finally {
        setIsLoading(false)
      }
    }

    fetchPredictions()
  }, [season, week, seasonType, bankroll])

  // Fetch results when results tab is active
  useEffect(() => {
    if (activeTab !== 'results') return

    const fetchResults = async () => {
      setResultsLoading(true)

      try {
        const data = await apiClient.getResults(season, week, seasonType, bankroll)
        setResults(data)
      } catch (err) {
        console.error('Failed to fetch results:', err)
        setResults(null)
      } finally {
        setResultsLoading(false)
      }
    }

    fetchResults()
  }, [activeTab, season, week, seasonType, bankroll])

  const handleRefresh = () => {
    // Re-fetch by triggering useEffect
    setSeason((s) => s)
  }

  // Filter and sort predictions
  const getFilteredPredictions = (): Prediction[] => {
    if (!predictions?.predictions) return []

    let filtered = [...predictions.predictions]

    // Apply filter
    switch (filter) {
      case 'bets':
        filtered = filtered.filter((p) => p.bet_recommendation === 'BET')
        break
      case 'leans':
        filtered = filtered.filter((p) => p.bet_recommendation === 'LEAN')
        break
      case 'actionable':
        filtered = filtered.filter((p) => p.bet_recommendation !== 'PASS')
        break
    }

    // Apply sort
    switch (sortBy) {
      case 'edge':
        filtered.sort((a, b) => Math.abs(b.predicted_edge) - Math.abs(a.predicted_edge))
        break
      case 'probability':
        filtered.sort((a, b) => b.cover_probability - a.cover_probability)
        break
    }

    return filtered
  }

  const filteredPredictions = getFilteredPredictions()

  // Separate top picks (BET recommendations) from the rest
  const topPicks = filteredPredictions
    .filter((p) => p.bet_recommendation === 'BET')
    .slice(0, 2)
  const morePicks = filteredPredictions.filter(
    (p) => !topPicks.includes(p)
  )

  // Calculate session summary metrics
  const calculateMetrics = () => {
    if (!filteredPredictions.length) return null

    const confidentPicks = filteredPredictions.filter(
      (p) => p.confidence_tier === 'HIGH' || p.confidence_tier === 'MEDIUM-HIGH'
    ).length

    const totalWagered = filteredPredictions
      .filter((p) => p.bet_recommendation !== 'PASS')
      .reduce((sum, p) => sum + Math.round(p.bet_size * bankroll), 0)

    const avgEdge =
      filteredPredictions.reduce((sum, p) => sum + Math.abs(p.predicted_edge), 0) /
      filteredPredictions.length

    const bestEdge = Math.max(...filteredPredictions.map((p) => Math.abs(p.predicted_edge)))

    return { confidentPicks, totalWagered, avgEdge, bestEdge }
  }

  const metrics = calculateMetrics()

  return (
    <div className="min-h-screen bg-slate-900">
      <Header onRefresh={handleRefresh} isLoading={isLoading} />

      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Week Selector */}
        <WeekSelector
          season={season}
          week={week}
          seasonType={seasonType}
          onSeasonChange={setSeason}
          onWeekChange={setWeek}
          onSeasonTypeChange={setSeasonType}
        />

        {/* Tab Navigation */}
        <div className="mt-4">
          <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
        </div>

        {/* Session Summary Metrics - 5 columns */}
        {predictions && metrics && activeTab === 'spreads' && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-slate-800 rounded-lg p-4">
              <p className="text-slate-400 text-sm">Total Games</p>
              <p className="text-2xl font-bold text-white">{filteredPredictions.length}</p>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <p className="text-slate-400 text-sm">Confident Picks</p>
              <p className="text-2xl font-bold text-emerald-400">{metrics.confidentPicks}</p>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <p className="text-slate-400 text-sm">Total Wagered</p>
              <p className="text-2xl font-bold text-emerald-400">${metrics.totalWagered}</p>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <p className="text-slate-400 text-sm">Avg Edge</p>
              <p className="text-2xl font-bold text-amber-400">{metrics.avgEdge.toFixed(1)} pts</p>
            </div>
            <div className="bg-slate-800 rounded-lg p-4">
              <p className="text-slate-400 text-sm">Best Edge</p>
              <p className="text-2xl font-bold text-emerald-400">{metrics.bestEdge.toFixed(1)} pts</p>
            </div>
          </div>
        )}

        {/* Filters and Controls */}
        {activeTab === 'spreads' && (
          <div className="mt-4 flex flex-wrap gap-4 items-center justify-between">
            <div className="flex gap-2">
              {(['all', 'actionable', 'bets', 'leans'] as FilterType[]).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                    filter === f
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {f === 'all' ? 'All Games' : f === 'actionable' ? 'BET + LEAN' : f.toUpperCase()}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400">Sort:</label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                  className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-white text-sm"
                >
                  <option value="default">Default</option>
                  <option value="edge">By Edge</option>
                  <option value="probability">By Probability</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400">Bankroll:</label>
                <input
                  type="number"
                  value={bankroll}
                  onChange={(e) => setBankroll(Number(e.target.value))}
                  className="w-24 bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-white text-sm"
                />
              </div>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="mt-6 bg-red-500/10 border border-red-500/50 rounded-lg p-4">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="mt-6 flex justify-center">
            <div className="flex items-center gap-3 text-slate-400">
              <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Loading predictions...
            </div>
          </div>
        )}

        {/* SPREADS TAB CONTENT */}
        {activeTab === 'spreads' && !isLoading && !error && (
          <>
            {/* Hero Section - Top Picks */}
            {topPicks.length > 0 && (
              <div className="mt-6">
                <HeroSection predictions={topPicks} bankroll={bankroll} />
              </div>
            )}

            {/* More Picks Section */}
            {morePicks.length > 0 && (
              <div className="mt-6">
                {topPicks.length > 0 && (
                  <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
                    More Picks ({morePicks.length})
                  </h2>
                )}
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {morePicks.map((prediction, idx) => (
                    <GameCard
                      key={prediction.game_id || idx}
                      prediction={prediction}
                      bankroll={bankroll}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Empty State */}
            {filteredPredictions.length === 0 && (
              <div className="mt-6 text-center py-12">
                <p className="text-slate-400">
                  {predictions?.total_games === 0
                    ? 'No games found for this week.'
                    : 'No games match your current filter.'}
                </p>
              </div>
            )}
          </>
        )}

        {/* TOTALS TAB CONTENT */}
        {activeTab === 'totals' && !isLoading && (
          <div className="mt-6 text-center py-12">
            <p className="text-slate-400">Totals (O/U) predictions coming soon...</p>
            <p className="text-slate-500 text-sm mt-2">
              This feature requires a backend endpoint to be added.
            </p>
          </div>
        )}

        {/* RESULTS TAB CONTENT */}
        {activeTab === 'results' && (
          <div className="mt-6">
            {resultsLoading ? (
              <div className="flex justify-center py-12">
                <div className="flex items-center gap-3 text-slate-400">
                  <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Loading results...
                </div>
              </div>
            ) : results ? (
              <>
                {/* Results Summary */}
                <ResultsSummary results={results} />

                {/* Results Table */}
                {results.results.length > 0 ? (
                  <div className="mt-6 bg-slate-800 rounded-lg overflow-hidden">
                    <ResultsTable results={results.results} />
                  </div>
                ) : (
                  <div className="mt-6 text-center py-12">
                    <p className="text-slate-400">
                      {results.status === 'no_games'
                        ? 'No completed games yet for this week. Check back after games finish!'
                        : results.status === 'no_lines'
                        ? 'No betting lines available for completed games.'
                        : 'No results to display.'}
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-12">
                <p className="text-slate-400">Failed to load results. Please try again.</p>
              </div>
            )}
          </div>
        )}

        {/* Last Refresh Info */}
        {predictions?.last_refresh && (
          <div className="mt-6 text-center">
            <p className="text-xs text-slate-500">
              Data last refreshed: {new Date(predictions.last_refresh).toLocaleString()}
            </p>
          </div>
        )}
      </main>
    </div>
  )
}
