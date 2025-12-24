'use client'

import { RECOMMENDATION_COLORS, CONFIDENCE_COLORS, BetRecommendation, ConfidenceTier } from '@/lib/types'

interface PredictionBadgeProps {
  recommendation: BetRecommendation
  size?: 'sm' | 'md' | 'lg'
}

export function PredictionBadge({ recommendation, size = 'md' }: PredictionBadgeProps) {
  const colors = RECOMMENDATION_COLORS[recommendation]

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  }

  return (
    <span
      className={`${sizeClasses[size]} font-bold rounded-full ${colors.bg} ${colors.text} ${colors.border || ''} border`}
    >
      {recommendation}
    </span>
  )
}

interface ConfidenceBadgeProps {
  tier: ConfidenceTier
  size?: 'sm' | 'md'
}

export function ConfidenceBadge({ tier, size = 'sm' }: ConfidenceBadgeProps) {
  const colors = CONFIDENCE_COLORS[tier]

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
  }

  return (
    <span
      className={`${sizeClasses[size]} font-medium rounded ${colors.bg} ${colors.text}`}
    >
      {tier}
    </span>
  )
}
