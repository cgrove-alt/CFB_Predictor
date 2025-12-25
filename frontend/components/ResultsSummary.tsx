'use client'

import { ResultsResponse } from '@/lib/types'

interface ResultsSummaryProps {
  results: ResultsResponse
}

export function ResultsSummary({ results }: ResultsSummaryProps) {
  const { wins, losses, win_rate, status, total_games } = results

  const getStatusDisplay = () => {
    switch (status) {
      case 'profitable':
        return { text: 'Profitable', emoji: '📈', color: 'text-emerald-400' }
      case 'break_even':
        return { text: 'Break Even', emoji: '📊', color: 'text-amber-400' }
      case 'review':
        return { text: 'Review', emoji: '📉', color: 'text-red-400' }
      case 'no_games':
        return { text: 'No Games', emoji: '⏳', color: 'text-slate-400' }
      case 'no_lines':
        return { text: 'No Lines', emoji: '❓', color: 'text-slate-400' }
      default:
        return { text: status, emoji: '📊', color: 'text-slate-400' }
    }
  }

  const statusDisplay = getStatusDisplay()

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="bg-slate-800 rounded-lg p-4">
        <p className="text-slate-400 text-sm">Games</p>
        <p className="text-2xl font-bold text-white">{total_games}</p>
      </div>
      <div className="bg-slate-800 rounded-lg p-4">
        <p className="text-slate-400 text-sm">Record</p>
        <p className="text-2xl font-bold">
          <span className="text-emerald-400">{wins}</span>
          <span className="text-slate-400"> - </span>
          <span className="text-red-400">{losses}</span>
        </p>
      </div>
      <div className="bg-slate-800 rounded-lg p-4">
        <p className="text-slate-400 text-sm">Win Rate</p>
        <p className={`text-2xl font-bold ${
          win_rate >= 55 ? 'text-emerald-400' :
          win_rate >= 50 ? 'text-amber-400' : 'text-red-400'
        }`}>
          {win_rate.toFixed(1)}%
        </p>
      </div>
      <div className="bg-slate-800 rounded-lg p-4">
        <p className="text-slate-400 text-sm">Status</p>
        <p className={`text-2xl font-bold ${statusDisplay.color}`}>
          {statusDisplay.emoji} {statusDisplay.text}
        </p>
      </div>
    </div>
  )
}
