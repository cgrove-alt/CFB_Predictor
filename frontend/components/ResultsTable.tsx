'use client'

import { GameResult } from '@/lib/types'
import { ConfidenceBadge } from './PredictionBadge'

interface ResultsTableProps {
  results: GameResult[]
}

export function ResultsTable({ results }: ResultsTableProps) {
  if (results.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-slate-400">No results to display</p>
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left py-3 px-4 text-slate-400 text-sm font-medium">Game</th>
            <th className="text-left py-3 px-4 text-slate-400 text-sm font-medium">Score</th>
            <th className="text-left py-3 px-4 text-slate-400 text-sm font-medium">Pick</th>
            <th className="text-center py-3 px-4 text-slate-400 text-sm font-medium">Result</th>
            <th className="text-right py-3 px-4 text-slate-400 text-sm font-medium">ATS Margin</th>
            <th className="text-center py-3 px-4 text-slate-400 text-sm font-medium">Confidence</th>
            <th className="text-right py-3 px-4 text-slate-400 text-sm font-medium">Bet Size</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, idx) => (
            <tr
              key={idx}
              className="border-b border-slate-700/50 hover:bg-slate-800/50 transition-colors"
            >
              <td className="py-3 px-4">
                <span className="text-white font-medium">{result.game}</span>
              </td>
              <td className="py-3 px-4">
                <span className="text-slate-300">
                  {result.away_score} - {result.home_score}
                </span>
              </td>
              <td className="py-3 px-4">
                <span className="text-slate-300">{result.pick}</span>
              </td>
              <td className="py-3 px-4 text-center">
                <span
                  className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                    result.result === 'WIN'
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}
                >
                  {result.result === 'WIN' ? '✓' : '✗'} {result.result}
                </span>
              </td>
              <td className="py-3 px-4 text-right">
                <span
                  className={`font-medium ${
                    result.ats_margin > 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {result.ats_margin > 0 ? '+' : ''}{result.ats_margin.toFixed(1)}
                </span>
              </td>
              <td className="py-3 px-4 text-center">
                <ConfidenceBadge tier={result.confidence_tier} />
              </td>
              <td className="py-3 px-4 text-right">
                {result.bet_recommendation !== 'PASS' && result.bet_size > 0 ? (
                  <span className="text-white font-medium">${result.bet_size}</span>
                ) : (
                  <span className="text-slate-500">-</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
