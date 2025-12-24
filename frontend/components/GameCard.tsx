'use client'

import { Prediction } from '@/lib/types'
import { PredictionBadge, ConfidenceBadge } from './PredictionBadge'

interface GameCardProps {
  prediction: Prediction
  bankroll: number
}

export function GameCard({ prediction, bankroll }: GameCardProps) {
  const {
    home_team,
    away_team,
    signal,
    team_to_bet,
    opponent,
    spread_to_bet,
    vegas_spread,
    predicted_margin,
    predicted_edge,
    cover_probability,
    bet_recommendation,
    confidence_tier,
    bet_size,
    kelly_fraction,
    line_movement,
    game_quality_score,
    start_date,
  } = prediction

  // Format date if available
  const formatDate = (dateStr: string | null | undefined) => {
    if (!dateStr) return null
    try {
      const date = new Date(dateStr)
      return date.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
      })
    } catch {
      return null
    }
  }

  const formattedDate = formatDate(start_date)
  const betAmount = Math.round(bet_size * bankroll)

  // Determine card border color based on recommendation
  const borderClass = {
    BET: 'border-emerald-500',
    LEAN: 'border-amber-500',
    PASS: 'border-slate-600',
  }[bet_recommendation] || 'border-slate-600'

  return (
    <div className={`bg-slate-800 rounded-lg border-2 ${borderClass} p-4 hover:bg-slate-750 transition-colors`}>
      {/* Header: Teams and Badge */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">
            {away_team} @ {home_team}
          </h3>
          {formattedDate && (
            <p className="text-sm text-slate-400 mt-1">{formattedDate}</p>
          )}
        </div>
        <PredictionBadge recommendation={bet_recommendation} size="lg" />
      </div>

      {/* Pick Info */}
      <div className="bg-slate-900/50 rounded-lg p-3 mb-4">
        <div className="flex justify-between items-center">
          <div>
            <span className="text-slate-400 text-sm">Pick: </span>
            <span className="text-white font-semibold">{team_to_bet}</span>
            <span className="text-slate-300 ml-2">
              {spread_to_bet > 0 ? '+' : ''}{spread_to_bet.toFixed(1)}
            </span>
          </div>
          <div className="text-right">
            <span className="text-slate-400 text-sm">vs </span>
            <span className="text-slate-300">{opponent}</span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Cover Probability */}
        <div className="bg-slate-900/30 rounded p-2">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Cover Prob</p>
          <p className={`text-xl font-bold ${
            cover_probability >= 0.55 ? 'text-emerald-400' :
            cover_probability >= 0.50 ? 'text-amber-400' : 'text-red-400'
          }`}>
            {(cover_probability * 100).toFixed(1)}%
          </p>
        </div>

        {/* Edge */}
        <div className="bg-slate-900/30 rounded p-2">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Edge</p>
          <p className={`text-xl font-bold ${
            predicted_edge >= 4.5 ? 'text-emerald-400' :
            predicted_edge >= 2.5 ? 'text-amber-400' : 'text-slate-300'
          }`}>
            {predicted_edge >= 0 ? '+' : ''}{predicted_edge.toFixed(1)} pts
          </p>
        </div>

        {/* Vegas Spread */}
        <div className="bg-slate-900/30 rounded p-2">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Vegas Line</p>
          <p className="text-lg font-semibold text-white">
            {vegas_spread > 0 ? '+' : ''}{vegas_spread.toFixed(1)}
          </p>
        </div>

        {/* Predicted Margin */}
        <div className="bg-slate-900/30 rounded p-2">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Model Pred</p>
          <p className="text-lg font-semibold text-white">
            {predicted_margin > 0 ? '+' : ''}{predicted_margin.toFixed(1)}
          </p>
        </div>
      </div>

      {/* Bottom Row: Confidence, Kelly, Line Movement */}
      <div className="flex flex-wrap gap-2 items-center justify-between border-t border-slate-700 pt-3">
        <div className="flex items-center gap-2">
          <ConfidenceBadge tier={confidence_tier} />
          {game_quality_score >= 4 && (
            <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">
              Quality: {game_quality_score}/5
            </span>
          )}
        </div>

        <div className="flex items-center gap-4 text-sm">
          {/* Line Movement */}
          {line_movement !== 0 && (
            <span className={`${
              line_movement > 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              Line: {line_movement > 0 ? '+' : ''}{line_movement.toFixed(1)}
            </span>
          )}

          {/* Kelly Bet Size */}
          {bet_recommendation !== 'PASS' && betAmount > 0 && (
            <span className="text-slate-300">
              Kelly: <span className="font-semibold text-white">${betAmount}</span>
              <span className="text-slate-500 ml-1">({(kelly_fraction * 100).toFixed(1)}%)</span>
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
