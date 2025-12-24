'use client'

import { RECOMMENDATION_COLORS, CONFIDENCE_COLORS } from '@/lib/types'

interface PredictionBadgeProps {
  recommendation: string
  size?: 'sm' | 'md' | 'lg'
}

export function PredictionBadge({ recommendation, size = 'md' }: PredictionBadgeProps) {
  const colors = RECOMMENDATION_COLORS[recommendation] || RECOMMENDATION_COLORS.PASS

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  }

  return (
    <span
      className={`${sizeClasses[size]} font-bold rounded-full ${colors.bg} ${colors.text} ${colors.border} border`}
    >
      {recommendation}
    </span>
  )
}

interface ConfidenceBadgeProps {
  tier: string
  size?: 'sm' | 'md'
}

export function ConfidenceBadge({ tier, size = 'sm' }: ConfidenceBadgeProps) {
  const colors = CONFIDENCE_COLORS[tier] || CONFIDENCE_COLORS.LOW

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
