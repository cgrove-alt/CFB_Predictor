'use client'

import { Prediction } from '@/lib/types'
import { PredictionBadge, ConfidenceBadge } from './PredictionBadge'
import { GameStatusBadge } from './GameStatusBadge'
import { EdgeTooltip } from './InfoTooltip'

interface HeroSectionProps {
  predictions: Prediction[]
}

export function HeroSection({ predictions }: HeroSectionProps) {
  if (predictions.length === 0) return null

  return (
    <div className="mb-8">
      {/* Section Header */}
      <div className="mb-4">
        <h2 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider">
          Top Picks
        </h2>
      </div>

      {/* Hero Cards Grid */}
      <div className={`grid gap-4 ${predictions.length === 1 ? 'grid-cols-1 max-w-xl' : 'grid-cols-1 md:grid-cols-2'}`}>
        {predictions.map((prediction, idx) => (
          <HeroCard
            key={prediction.game_id || idx}
            prediction={prediction}
          />
        ))}
      </div>
    </div>
  )
}

interface HeroCardProps {
  prediction: Prediction
}

function HeroCard({ prediction }: HeroCardProps) {
  const {
    team_to_bet,
    opponent,
    spread_to_bet,
    vegas_spread,
    predicted_margin,
    predicted_edge,
    bet_recommendation,
    confidence_tier,
    line_movement,
    start_date,
    signal,
  } = prediction

  // Border color based on signal (BUY = green, FADE = red)
  const borderColor = signal === 'BUY' ? 'border-l-emerald-500' : 'border-l-red-500'

  return (
    <div className={`bg-gradient-to-br from-slate-800 to-slate-850 rounded-xl border-l-4 ${borderColor} p-6 shadow-lg`}>
      {/* Top Row: Badges */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex gap-2">
          <ConfidenceBadge tier={confidence_tier} />
          <PredictionBadge recommendation={bet_recommendation} size="lg" />
        </div>
        {start_date && <GameStatusBadge startDate={start_date} />}
      </div>

      {/* Main Bet Instruction - Large and Prominent */}
      <div className="mb-4">
        <p className="text-3xl font-bold text-white">
          {bet_recommendation}: {team_to_bet}{' '}
          <span className="text-emerald-400">
            {spread_to_bet > 0 ? '+' : ''}{spread_to_bet.toFixed(1)}
          </span>
        </p>
        <p className="text-slate-400 mt-1">
          vs {opponent}
        </p>
      </div>

      {/* Edge and Stats */}
      <div className="border-t border-slate-700 pt-4">
        <p className="text-amber-400 text-sm mb-2 flex items-center">
          Predicted edge: {predicted_edge >= 0 ? '+' : ''}{predicted_edge.toFixed(1)} pts vs Vegas
          <EdgeTooltip />
        </p>

        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <p className="text-slate-500 text-xs uppercase">Vegas</p>
            <p className="text-white font-medium">
              {vegas_spread > 0 ? '+' : ''}{vegas_spread.toFixed(1)}
            </p>
          </div>
          <div>
            <p className="text-slate-500 text-xs uppercase">Model</p>
            <p className="text-white font-medium">
              {predicted_margin > 0 ? '+' : ''}{predicted_margin.toFixed(1)}
            </p>
          </div>
          {line_movement !== 0 && (
            <div>
              <p className="text-slate-500 text-xs uppercase">Movement</p>
              <p className={`font-medium ${line_movement > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {line_movement > 0 ? '+' : ''}{line_movement.toFixed(1)}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
