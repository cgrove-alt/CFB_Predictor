'use client'

import { useEffect, useState } from 'react'

interface GameStatusBadgeProps {
  startDate: string
}

type StatusType = 'live' | 'soon' | 'today' | 'future' | 'completed'

interface GameStatus {
  text: string
  type: StatusType
}

function getGameStatus(startDate: string): GameStatus {
  const gameTime = new Date(startDate)
  const now = new Date()
  const diffMs = gameTime.getTime() - now.getTime()
  const diffHours = diffMs / (1000 * 60 * 60)
  const diffMinutes = diffMs / (1000 * 60)

  // Game is in progress (started within last 4 hours)
  if (diffMs < 0 && diffMs > -4 * 60 * 60 * 1000) {
    return { text: 'LIVE', type: 'live' }
  }

  // Game already completed (more than 4 hours ago)
  if (diffMs < -4 * 60 * 60 * 1000) {
    return { text: 'Final', type: 'completed' }
  }

  // Less than 1 hour until kickoff
  if (diffHours < 1 && diffHours >= 0) {
    return { text: `Kickoff in ${Math.floor(diffMinutes)}m`, type: 'soon' }
  }

  // Less than 24 hours
  if (diffHours < 24 && diffHours >= 1) {
    const hours = Math.floor(diffHours)
    const mins = Math.floor((diffMs % (60 * 60 * 1000)) / (60 * 1000))
    return { text: `Kickoff in ${hours}h ${mins}m`, type: 'today' }
  }

  // Future games
  const formatted = gameTime.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
  return { text: formatted, type: 'future' }
}

export function GameStatusBadge({ startDate }: GameStatusBadgeProps) {
  const [status, setStatus] = useState<GameStatus>(() => getGameStatus(startDate))

  // Update status every minute for countdown
  useEffect(() => {
    const interval = setInterval(() => {
      setStatus(getGameStatus(startDate))
    }, 60000) // Update every minute

    return () => clearInterval(interval)
  }, [startDate])

  const styleClasses: Record<StatusType, string> = {
    live: 'bg-red-500 text-white animate-pulse',
    soon: 'bg-amber-500 text-slate-900',
    today: 'bg-blue-500 text-white',
    future: 'bg-slate-600 text-slate-200',
    completed: 'bg-slate-700 text-slate-400',
  }

  return (
    <span className={`text-xs px-2 py-1 rounded-full font-medium ${styleClasses[status.type]}`}>
      {status.text}
    </span>
  )
}
