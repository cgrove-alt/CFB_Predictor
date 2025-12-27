'use client'

import { Prediction } from '@/lib/types'
import { GameStatusBadge } from './GameStatusBadge'
import { EdgeTooltip } from './InfoTooltip'

interface GameCardProps {
  prediction: Prediction
}

// Confidence tier to percentage for progress bar
const CONFIDENCE_TO_PERCENT: Record<string, number> = {
  'HIGH': 95,
  'MEDIUM-HIGH': 75,
  'MEDIUM': 55,
  'LOW': 35,
  'VERY LOW': 15,
}

export function GameCard({ prediction }: GameCardProps) {
  const {
    home_team,
    away_team,
    team_to_bet,
    vegas_spread,
    predicted_edge,
    bet_recommendation,
    confidence_tier,
    start_date,
  } = prediction

  const confidencePercent = CONFIDENCE_TO_PERCENT[confidence_tier] || 50

  // Determine which side to highlight (home or away)
  const homeIsRecommended = team_to_bet === home_team
  const awayIsRecommended = team_to_bet === away_team

  // Calculate spreads for each side
  const homeSpread = vegas_spread
  const awaySpread = -vegas_spread

  // Edge color coding
  const getEdgeColor = (edge: number) => {
    const absEdge = Math.abs(edge)
    if (absEdge >= 3) return 'text-emerald-400'
    if (absEdge >= 1) return 'text-amber-400'
    return 'text-slate-400'
  }

  // Format date/time
  const formatGameTime = (dateStr?: string) => {
    if (!dateStr) return null
    const date = new Date(dateStr)
    const now = new Date()
    const isToday = date.toDateString() === now.toDateString()
    const tomorrow = new Date(now)
    tomorrow.setDate(tomorrow.getDate() + 1)
    const isTomorrow = date.toDateString() === tomorrow.toDateString()

    const timeStr = date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })

    if (isToday) return `Today ${timeStr}`
    if (isTomorrow) return `Tomorrow ${timeStr}`

    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    })
  }

  // Get confidence bar color
  const getConfidenceBarColor = () => {
    if (confidence_tier === 'HIGH') return 'bg-emerald-500'
    if (confidence_tier === 'MEDIUM-HIGH') return 'bg-emerald-400'
    if (confidence_tier === 'MEDIUM') return 'bg-amber-500'
    if (confidence_tier === 'LOW') return 'bg-amber-400'
    return 'bg-red-500'
  }

  return (
    <div className="bg-slate-800 rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-shadow border border-slate-700/50">
      {/* Header: Time & Date */}
      <div className="px-4 py-2.5 bg-slate-900/50 border-b border-slate-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 font-medium">
            {formatGameTime(start_date) || 'TBD'}
          </span>
          {start_date && (
            <GameStatusBadge startDate={start_date} />
          )}
        </div>
        {/* Optional: TV Station placeholder */}
        <span className="text-xs text-slate-500">CFB</span>
      </div>

      {/* Main Content */}
      <div className="p-4">
        {/* Teams & Betting Buttons Row */}
        <div className="flex items-stretch gap-3">
          {/* Teams Column */}
          <div className="flex-1 min-w-0">
            {/* Away Team */}
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-bold text-slate-300 flex-shrink-0">
                {away_team.substring(0, 2).toUpperCase()}
              </div>
              <span className="font-semibold text-white truncate">{away_team}</span>
            </div>

            {/* Home Team */}
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-bold text-slate-300 flex-shrink-0">
                {home_team.substring(0, 2).toUpperCase()}
              </div>
              <span className="font-semibold text-white truncate">{home_team}</span>
            </div>
          </div>

          {/* Betting Buttons Column */}
          <div className="flex flex-col gap-2 w-28 flex-shrink-0">
            {/* Away Spread Button */}
            <button
              className={`relative flex items-center justify-center px-3 py-2.5 rounded-lg font-bold text-sm transition-all ${
                awayIsRecommended && bet_recommendation !== 'PASS'
                  ? 'bg-emerald-600 text-white ring-2 ring-emerald-400 shadow-lg shadow-emerald-500/20'
                  : 'bg-slate-700/80 text-slate-300 hover:bg-slate-600/80'
              }`}
            >
              {awayIsRecommended && bet_recommendation !== 'PASS' && (
                <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              )}
              <span>{awaySpread > 0 ? '+' : ''}{awaySpread.toFixed(1)}</span>
            </button>

            {/* Home Spread Button */}
            <button
              className={`relative flex items-center justify-center px-3 py-2.5 rounded-lg font-bold text-sm transition-all ${
                homeIsRecommended && bet_recommendation !== 'PASS'
                  ? 'bg-emerald-600 text-white ring-2 ring-emerald-400 shadow-lg shadow-emerald-500/20'
                  : 'bg-slate-700/80 text-slate-300 hover:bg-slate-600/80'
              }`}
            >
              {homeIsRecommended && bet_recommendation !== 'PASS' && (
                <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              )}
              <span>{homeSpread > 0 ? '+' : ''}{homeSpread.toFixed(1)}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Footer Stats Bar */}
      <div className="px-4 py-3 bg-slate-900/30 border-t border-slate-700/50">
        <div className="flex items-center justify-between gap-4">
          {/* Edge */}
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-500 uppercase tracking-wide">Edge</span>
            <EdgeTooltip />
            <span className={`text-sm font-bold ml-1 ${getEdgeColor(predicted_edge)}`}>
              {predicted_edge >= 0 ? '+' : ''}{predicted_edge.toFixed(1)} pts
            </span>
          </div>

          {/* Confidence Bar */}
          <div className="flex items-center gap-2 flex-1 max-w-[140px]">
            <span className="text-xs text-slate-500 uppercase tracking-wide whitespace-nowrap">Conf</span>
            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full ${getConfidenceBarColor()} transition-all duration-500`}
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
          </div>

          {/* Recommendation Badge */}
          <span className={`text-xs font-semibold uppercase px-2 py-0.5 rounded ${
            bet_recommendation === 'BET'
              ? 'bg-emerald-500/20 text-emerald-400'
              : bet_recommendation === 'LEAN'
              ? 'bg-blue-500/20 text-blue-400'
              : 'bg-slate-600/50 text-slate-400'
          }`}>
            {bet_recommendation}
          </span>
        </div>
      </div>
    </div>
  )
}
