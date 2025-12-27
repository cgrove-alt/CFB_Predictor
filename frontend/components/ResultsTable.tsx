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
        <thead className="sticky top-0 bg-slate-800 z-10">
          <tr className="border-b border-slate-700">
            <th className="text-left py-3 px-4 text-slate-400 text-sm font-medium bg-slate-800">
              Game
            </th>
            <th className="text-left py-3 px-4 text-slate-400 text-sm font-medium bg-slate-800">
              Score
            </th>
            <th className="text-left py-3 px-4 text-slate-400 text-sm font-medium bg-slate-800">
              Pick
            </th>
            <th className="text-center py-3 px-4 text-slate-400 text-sm font-medium bg-slate-800">
              Result
            </th>
            <th className="text-right py-3 px-4 text-slate-400 text-sm font-medium bg-slate-800">
              ATS Margin
            </th>
            <th className="text-center py-3 px-4 text-slate-400 text-sm font-medium bg-slate-800">
              Confidence
            </th>
            <th className="text-right py-3 px-4 text-slate-400 text-sm font-medium bg-slate-800">
              Bet Size
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, idx) => (
            <tr
              key={idx}
              className={`border-b border-slate-700/50 hover:bg-slate-800/50 transition-colors ${
                result.result === 'WIN'
                  ? 'bg-emerald-500/5'
                  : 'bg-red-500/5'
              }`}
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
                <div className="flex items-center gap-2">
                  <span className="text-slate-300">{result.pick}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded ${
                    result.signal === 'BUY'
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {result.signal}
                  </span>
                </div>
              </td>
              <td className="py-3 px-4 text-center">
                <span
                  className={`inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-bold ${
                    result.result === 'WIN'
                      ? 'bg-emerald-500/30 text-emerald-300 ring-1 ring-emerald-500/50'
                      : 'bg-red-500/30 text-red-300 ring-1 ring-red-500/50'
                  }`}
                >
                  {result.result === 'WIN' ? (
                    <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  )}
                  {result.result}
                </span>
              </td>
              <td className="py-3 px-4 text-right">
                <span
                  className={`font-semibold ${
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
                  <span className={`font-semibold ${
                    result.result === 'WIN' ? 'text-emerald-400' : 'text-red-400'
                  }`}>
                    {result.result === 'WIN' ? '+' : '-'}${result.bet_size}
                  </span>
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

/**
 * ResultsTableSkeleton - Loading placeholder for results table
 */
export function ResultsTableSkeleton() {
  return (
    <div className="overflow-x-auto animate-pulse">
      <table className="w-full">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left py-3 px-4"><div className="h-4 w-16 bg-slate-700 rounded" /></th>
            <th className="text-left py-3 px-4"><div className="h-4 w-12 bg-slate-700 rounded" /></th>
            <th className="text-left py-3 px-4"><div className="h-4 w-10 bg-slate-700 rounded" /></th>
            <th className="text-center py-3 px-4"><div className="h-4 w-14 bg-slate-700 rounded mx-auto" /></th>
            <th className="text-right py-3 px-4"><div className="h-4 w-20 bg-slate-700 rounded ml-auto" /></th>
            <th className="text-center py-3 px-4"><div className="h-4 w-20 bg-slate-700 rounded mx-auto" /></th>
            <th className="text-right py-3 px-4"><div className="h-4 w-16 bg-slate-700 rounded ml-auto" /></th>
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: 5 }).map((_, idx) => (
            <tr key={idx} className="border-b border-slate-700/50">
              <td className="py-3 px-4"><div className="h-4 w-40 bg-slate-700 rounded" /></td>
              <td className="py-3 px-4"><div className="h-4 w-16 bg-slate-700 rounded" /></td>
              <td className="py-3 px-4"><div className="h-4 w-24 bg-slate-700 rounded" /></td>
              <td className="py-3 px-4 text-center"><div className="h-6 w-14 bg-slate-700 rounded mx-auto" /></td>
              <td className="py-3 px-4 text-right"><div className="h-4 w-12 bg-slate-700 rounded ml-auto" /></td>
              <td className="py-3 px-4 text-center"><div className="h-5 w-20 bg-slate-700 rounded mx-auto" /></td>
              <td className="py-3 px-4 text-right"><div className="h-4 w-12 bg-slate-700 rounded ml-auto" /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
