'use client'

/**
 * GameCardSkeleton - Loading placeholder for GameCard
 * Matches the exact dimensions and layout of the new GameCard component
 */
export function GameCardSkeleton() {
  return (
    <div className="bg-slate-800 rounded-xl overflow-hidden shadow-lg border border-slate-700/50 animate-pulse">
      {/* Header: Time & Date skeleton */}
      <div className="px-4 py-2.5 bg-slate-900/50 border-b border-slate-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-3 w-24 bg-slate-700 rounded" />
          <div className="h-4 w-12 bg-slate-700 rounded-full" />
        </div>
        <div className="h-3 w-8 bg-slate-700 rounded" />
      </div>

      {/* Main Content */}
      <div className="p-4">
        {/* Teams & Betting Buttons Row */}
        <div className="flex items-stretch gap-3">
          {/* Teams Column */}
          <div className="flex-1 min-w-0">
            {/* Away Team skeleton */}
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-full bg-slate-700 flex-shrink-0" />
              <div className="h-4 w-32 bg-slate-700 rounded" />
            </div>

            {/* Home Team skeleton */}
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-slate-700 flex-shrink-0" />
              <div className="h-4 w-28 bg-slate-700 rounded" />
            </div>
          </div>

          {/* Betting Buttons Column skeleton */}
          <div className="flex flex-col gap-2 w-28 flex-shrink-0">
            <div className="h-10 bg-slate-700 rounded-lg" />
            <div className="h-10 bg-slate-700 rounded-lg" />
          </div>
        </div>
      </div>

      {/* Footer Stats Bar skeleton */}
      <div className="px-4 py-3 bg-slate-900/30 border-t border-slate-700/50">
        <div className="flex items-center justify-between gap-4">
          {/* Edge skeleton */}
          <div className="flex items-center gap-1">
            <div className="h-3 w-8 bg-slate-700 rounded" />
            <div className="h-4 w-16 bg-slate-700 rounded ml-1" />
          </div>

          {/* Kelly skeleton */}
          <div className="flex items-center gap-1">
            <div className="h-3 w-8 bg-slate-700 rounded" />
            <div className="h-4 w-10 bg-slate-700 rounded ml-1" />
          </div>

          {/* Confidence Bar skeleton */}
          <div className="flex items-center gap-2 flex-1 max-w-[120px]">
            <div className="h-3 w-8 bg-slate-700 rounded" />
            <div className="flex-1 h-2 bg-slate-700 rounded-full" />
          </div>
        </div>

        {/* Bet Amount Row skeleton */}
        <div className="mt-2 pt-2 border-t border-slate-700/50 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-5 w-12 bg-slate-700 rounded" />
            <div className="h-3 w-20 bg-slate-700 rounded" />
          </div>
          <div className="h-6 w-16 bg-slate-700 rounded" />
        </div>
      </div>
    </div>
  )
}

/**
 * GameCardSkeletonGrid - Multiple skeletons for loading state
 */
interface GameCardSkeletonGridProps {
  count?: number
}

export function GameCardSkeletonGrid({ count = 6 }: GameCardSkeletonGridProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: count }).map((_, idx) => (
        <GameCardSkeleton key={idx} />
      ))}
    </div>
  )
}

/**
 * HeroSkeleton - Loading placeholder for hero section
 */
export function HeroSkeleton() {
  return (
    <div className="mb-8">
      {/* Section Header skeleton */}
      <div className="mb-4">
        <div className="h-4 w-24 bg-slate-700 rounded animate-pulse" />
      </div>

      {/* Hero Cards Grid */}
      <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
        <HeroCardSkeleton />
        <HeroCardSkeleton />
      </div>
    </div>
  )
}

/**
 * HeroCardSkeleton - Single hero card loading placeholder
 */
function HeroCardSkeleton() {
  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-850 rounded-xl border-l-4 border-l-slate-600 p-6 shadow-lg animate-pulse">
      {/* Top Row: Badges skeleton */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex gap-2">
          <div className="h-6 w-20 bg-slate-700 rounded-full" />
          <div className="h-6 w-16 bg-slate-700 rounded-full" />
        </div>
        <div className="h-5 w-24 bg-slate-700 rounded-full" />
      </div>

      {/* Main Bet Instruction skeleton */}
      <div className="mb-4">
        <div className="h-8 w-64 bg-slate-700 rounded mb-2" />
        <div className="h-4 w-24 bg-slate-700 rounded" />
      </div>

      {/* Bet Amount skeleton */}
      <div className="mb-4">
        <div className="h-10 w-24 bg-slate-700 rounded mb-1" />
        <div className="h-3 w-32 bg-slate-700 rounded" />
      </div>

      {/* Edge and Stats skeleton */}
      <div className="border-t border-slate-700 pt-4">
        <div className="h-4 w-48 bg-slate-700 rounded mb-3" />
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="h-2 w-12 bg-slate-700 rounded mb-1" />
            <div className="h-4 w-10 bg-slate-700 rounded" />
          </div>
          <div>
            <div className="h-2 w-12 bg-slate-700 rounded mb-1" />
            <div className="h-4 w-10 bg-slate-700 rounded" />
          </div>
          <div>
            <div className="h-2 w-16 bg-slate-700 rounded mb-1" />
            <div className="h-4 w-10 bg-slate-700 rounded" />
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * MetricsSkeleton - Loading placeholder for session metrics
 */
export function MetricsSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 animate-pulse">
      {Array.from({ length: 5 }).map((_, idx) => (
        <div key={idx} className="bg-slate-800 rounded-lg p-4">
          <div className="h-3 w-20 bg-slate-700 rounded mb-2" />
          <div className="h-7 w-16 bg-slate-700 rounded" />
        </div>
      ))}
    </div>
  )
}
