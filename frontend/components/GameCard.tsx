'use client'

import { Prediction } from '@/lib/types'
import { PredictionBadge, ConfidenceBadge } from './PredictionBadge'
import { GameStatusBadge } from './GameStatusBadge'
import { EdgeTooltip } from './InfoTooltip'

interface GameCardProps {
  prediction: Prediction
  bankroll: number
  isHero?: boolean
}

export function GameCard({ prediction, bankroll, isHero = false }: GameCardProps) {
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

  const betAmount = Math.round(bet_size * bankroll)

  // Border color based on signal (BUY = green, FADE = red)
  const borderColor = signal === 'BUY' ? 'border-l-emerald-500' : 'border-l-red-500'

  return (
    <div className={`bg-slate-800 rounded-lg border-l-4 ${borderColor} p-4 hover:bg-slate-750 transition-colors`}>
      {/* Header: Teams, Date, and Badge */}
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className={`font-semibold text-white ${isHero ? 'text-xl' : 'text-lg'}`}>
            {away_team} @ {home_team}
          </h3>
          {start_date && (
            <div className="mt-1">
              <GameStatusBadge startDate={start_date} />
            </div>
          )}
        </div>
        <PredictionBadge recommendation={bet_recommendation} size="lg" />
      </div>

      {/* Bet Instruction - Prominent Display */}
      <div className="bg-slate-900/50 rounded-lg p-3 mb-3">
        <p className={`font-bold text-white ${isHero ? 'text-2xl' : 'text-xl'}`}>
          {bet_recommendation}: {team_to_bet}{' '}
          <span className="text-emerald-400">
            {spread_to_bet > 0 ? '+' : ''}{spread_to_bet.toFixed(1)}
          </span>
        </p>
        <p className="text-slate-400 text-sm mt-1">
          vs {opponent}
        </p>
      </div>

      {/* Bet Amount - Large Green Display */}
      {bet_recommendation !== 'PASS' && betAmount > 0 && (
        <div className="mb-3">
          <p className={`font-bold text-emerald-400 ${isHero ? 'text-4xl' : 'text-3xl'}`}>
            ${betAmount}
          </p>
          <p className="text-sm text-slate-400">
            {(cover_probability * 100).toFixed(0)}% Est. Win Probability
          </p>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        {/* Cover Probability */}
        <div className="bg-slate-900/30 rounded p-2">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Cover Prob</p>
          <p className={`text-lg font-bold ${
            cover_probability >= 0.55 ? 'text-emerald-400' :
            cover_probability >= 0.50 ? 'text-amber-400' : 'text-red-400'
          }`}>
            {(cover_probability * 100).toFixed(1)}%
          </p>
        </div>

        {/* Edge */}
        <div className="bg-slate-900/30 rounded p-2">
          <p className="text-xs text-slate-400 uppercase tracking-wide flex items-center">
            Edge <EdgeTooltip />
          </p>
          <p className={`text-lg font-bold ${
            Math.abs(predicted_edge) >= 4.5 ? 'text-emerald-400' :
            Math.abs(predicted_edge) >= 2.5 ? 'text-amber-400' : 'text-slate-300'
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

      {/* Bottom Row: Confidence, Quality, Line Movement */}
      <div className="flex flex-wrap gap-2 items-center justify-between border-t border-slate-700 pt-3">
        <div className="flex items-center gap-2">
          <ConfidenceBadge tier={confidence_tier} />
          {game_quality_score >= 4 && (
            <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">
              Quality: {game_quality_score}/5
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-sm">
          {/* Line Movement */}
          {line_movement !== 0 && (
            <span className={`${
              line_movement > 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              Line: {line_movement > 0 ? '+' : ''}{line_movement.toFixed(1)}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
