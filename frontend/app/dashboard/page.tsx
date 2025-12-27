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
import { ResultsTable, ResultsTableSkeleton } from '@/components/ResultsTable'
import { ResultsSummary } from '@/components/ResultsSummary'
import { SortControl, SortOption } from '@/components/SortControl'
import { GameCardSkeletonGrid, HeroSkeleton, MetricsSkeleton } from '@/components/GameCardSkeleton'

type FilterType = 'all' | 'bets' | 'leans' | 'actionable'

// Confidence tier ranking for sorting
const CONFIDENCE_RANK: Record<string, number> = {
  'HIGH': 5,
  'MEDIUM-HIGH': 4,
  'MEDIUM': 3,
  'LOW': 2,
  'VERY LOW': 1,
}

// Calculate initial season type based on current date
// This runs synchronously BEFORE render to avoid race conditions
const getInitialSeasonType = (): string => {
  const now = new Date()
  // December 15+ or January 1-10 = bowl season
  if ((now.getMonth() === 11 && now.getDate() >= 15) ||
      (now.getMonth() === 0 && now.getDate() <= 10)) {
    return 'postseason'
  }
  return 'regular'
}

// Calculate initial week based on current date
const getInitialWeek = (): number => {
  const now = new Date()
  const currentYear = now.getFullYear()

  // Bowl season = week 1
  if ((now.getMonth() === 11 && now.getDate() >= 15) ||
      (now.getMonth() === 0 && now.getDate() <= 10)) {
    return 1
  }

  // Regular season calculation
  const seasonStart = new Date(currentYear, 7, 24) // Late August
  if (now >= seasonStart) {
    const weeksSinceStart = Math.floor((now.getTime() - seasonStart.getTime()) / (7 * 24 * 60 * 60 * 1000))
    return Math.min(Math.max(weeksSinceStart + 1, 1), 15)
  }
  return 1
}

export default function DashboardPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Tab state
  const [activeTab, setActiveTab] = useState<TabType>('spreads')

  // Selection state - use smart initial values based on current date
  const currentYear = new Date().getFullYear()
  const [season, setSeason] = useState(currentYear)
  const [week, setWeek] = useState(getInitialWeek)
  const [seasonType, setSeasonType] = useState(getInitialSeasonType)

  // Data state
  const [predictions, setPredictions] = useState<PredictionsResponse | null>(null)
  const [results, setResults] = useState<ResultsResponse | null>(null)
  const [resultsLoading, setResultsLoading] = useState(false)

  // Filter state
  const [filter, setFilter] = useState<FilterType>('all')
  const [sortBy, setSortBy] = useState<SortOption>('edge')

  // Check authentication
  useEffect(() => {
    const isAuth = localStorage.getItem('authenticated')
    if (isAuth !== 'true') {
      router.push('/')
    }
    // Note: Week and seasonType are now calculated at initialization time
    // via getInitialWeek() and getInitialSeasonType() to avoid race conditions
  }, [router])

  // Fetch predictions when selection changes
  useEffect(() => {
    const fetchPredictions = async () => {
      setIsLoading(true)
      setError(null)

      try {
        const data = await apiClient.getPredictions(season, week, seasonType)
        setPredictions(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch predictions')
      } finally {
        setIsLoading(false)
      }
    }

    fetchPredictions()
  }, [season, week, seasonType])

  // Fetch results when results tab is active
  useEffect(() => {
    if (activeTab !== 'results') return

    const fetchResults = async () => {
      setResultsLoading(true)

      try {
        const data = await apiClient.getResults(season, week, seasonType)
        setResults(data)
      } catch (err) {
        console.error('Failed to fetch results:', err)
        setResults(null)
      } finally {
        setResultsLoading(false)
      }
    }

    fetchResults()
  }, [activeTab, season, week, seasonType])

  const handleRefresh = () => {
    // Re-fetch by triggering useEffect
    setSeason((s) => s)
  }

  // Filter and sort predictions
  const getFilteredPredictions = (): Prediction[] => {
    if (!predictions?.predictions) return []

    // Start with only upcoming games (not completed)
    let filtered = predictions.predictions.filter((p) => !p.completed)

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
      case 'confidence':
        filtered.sort((a, b) =>
          (CONFIDENCE_RANK[b.confidence_tier] || 0) - (CONFIDENCE_RANK[a.confidence_tier] || 0)
        )
        break
      case 'time':
        filtered.sort((a, b) => {
          if (!a.start_date && !b.start_date) return 0
          if (!a.start_date) return 1
          if (!b.start_date) return -1
          return new Date(a.start_date).getTime() - new Date(b.start_date).getTime()
        })
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

    const avgEdge =
      filteredPredictions.reduce((sum, p) => sum + Math.abs(p.predicted_edge), 0) /
      filteredPredictions.length

    const bestEdge = Math.max(...filteredPredictions.map((p) => Math.abs(p.predicted_edge)))

    return { confidentPicks, avgEdge, bestEdge }
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

        {/* Session Summary Metrics */}
        {activeTab === 'spreads' && (
          <div className="mt-4">
            {isLoading ? (
              <MetricsSkeleton />
            ) : predictions && metrics ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-800 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Total Games</p>
                  <p className="text-2xl font-bold text-white">{filteredPredictions.length}</p>
                </div>
                <div className="bg-slate-800 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">Confident Picks</p>
                  <p className="text-2xl font-bold text-emerald-400">{metrics.confidentPicks}</p>
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
            ) : null}
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

            {/* Sort Control */}
            <SortControl value={sortBy} onChange={setSortBy} />
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="mt-6 bg-red-500/10 border border-red-500/50 rounded-lg p-4">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* SPREADS TAB CONTENT */}
        {activeTab === 'spreads' && (
          <>
            {/* Loading State with Skeletons */}
            {isLoading && (
              <div className="mt-6">
                <HeroSkeleton />
                <div className="mt-6">
                  <div className="h-4 w-32 bg-slate-700 rounded animate-pulse mb-4" />
                  <GameCardSkeletonGrid count={6} />
                </div>
              </div>
            )}

            {/* Loaded Content */}
            {!isLoading && !error && (
              <>
                {/* Hero Section - Top Picks */}
                {topPicks.length > 0 && (
                  <div className="mt-6">
                    <HeroSection predictions={topPicks} />
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
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Empty State */}
                {filteredPredictions.length === 0 && (
                  <div className="mt-6 text-center py-12">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-800 mb-4">
                      <svg className="w-8 h-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
                      </svg>
                    </div>
                    <p className="text-slate-400 text-lg">
                      {predictions?.total_games === 0
                        ? 'No games found for this week.'
                        : 'No games match your current filter.'}
                    </p>
                    <p className="text-slate-500 text-sm mt-2">
                      Try changing your filters or selecting a different week.
                    </p>
                  </div>
                )}
              </>
            )}
          </>
        )}

        {/* TOTALS TAB CONTENT */}
        {activeTab === 'totals' && !isLoading && (
          <div className="mt-6 text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-800 mb-4">
              <svg className="w-8 h-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-slate-400 text-lg">Totals (O/U) predictions coming soon...</p>
            <p className="text-slate-500 text-sm mt-2">
              This feature requires a backend endpoint to be added.
            </p>
          </div>
        )}

        {/* RESULTS TAB CONTENT */}
        {activeTab === 'results' && (
          <div className="mt-6">
            {resultsLoading ? (
              <div className="space-y-6">
                {/* Summary Skeleton */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-pulse">
                  {Array.from({ length: 4 }).map((_, idx) => (
                    <div key={idx} className="bg-slate-800 rounded-lg p-4">
                      <div className="h-3 w-16 bg-slate-700 rounded mb-2" />
                      <div className="h-8 w-12 bg-slate-700 rounded" />
                    </div>
                  ))}
                </div>
                {/* Table Skeleton */}
                <div className="bg-slate-800 rounded-lg overflow-hidden">
                  <ResultsTableSkeleton />
                </div>
              </div>
            ) : results ? (
              <>
                {/* Results Summary */}
                <ResultsSummary results={results} />

                {/* Results Table */}
                {results.results.length > 0 ? (
                  <div className="mt-6 bg-slate-800 rounded-lg overflow-hidden max-h-[600px] overflow-y-auto">
                    <ResultsTable results={results.results} />
                  </div>
                ) : (
                  <div className="mt-6 text-center py-12">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-800 mb-4">
                      <svg className="w-8 h-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25z" />
                      </svg>
                    </div>
                    <p className="text-slate-400 text-lg">
                      {results.status === 'no_games'
                        ? 'No completed games yet for this week.'
                        : results.status === 'no_lines'
                        ? 'No betting lines available for completed games.'
                        : 'No results to display.'}
                    </p>
                    <p className="text-slate-500 text-sm mt-2">
                      Check back after games finish!
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
