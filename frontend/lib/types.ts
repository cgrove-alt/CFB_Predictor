// API Response Types

export interface Prediction {
  home_team: string;
  away_team: string;
  game: string;
  signal: 'BUY' | 'FADE';
  team_to_bet: string;
  opponent: string;
  spread_to_bet: number;
  vegas_spread: number;
  predicted_margin: number;
  predicted_edge: number;
  cover_probability: number;
  bet_recommendation: 'BET' | 'LEAN' | 'PASS';
  confidence_tier: 'HIGH' | 'MEDIUM-HIGH' | 'MEDIUM' | 'LOW' | 'VERY LOW';
  bet_size: number;
  kelly_fraction: number;
  line_movement: number;
  game_quality_score: number;
  start_date?: string;
  completed: boolean;
  game_id?: number;
}

export interface PredictionsResponse {
  season: number;
  week: number;
  season_type: string;
  last_refresh?: string;
  predictions: Prediction[];
  total_games: number;
  bet_count: number;
  lean_count: number;
  pass_count: number;
}

export interface StatusResponse {
  status: string;
  last_refresh?: string;
  next_refresh?: string;
  is_gameday: boolean;
  interval_hours: number;
  model_loaded: boolean;
  data_loaded: boolean;
  games_in_history: number;
}

export interface AuthResponse {
  authenticated: boolean;
  message: string;
}

export interface Game {
  id: number;
  home_team: string;
  away_team: string;
  start_date?: string;
  completed: boolean;
  venue?: string;
  home_points?: number;
  away_points?: number;
  vegas_spread?: number;
  over_under?: number;
}

export interface GamesResponse {
  season: number;
  week: number;
  season_type: string;
  games: Game[];
}

// Results Types
export interface GameResult {
  game: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  pick: string;
  signal: 'BUY' | 'FADE';
  spread_to_bet: number;
  result: 'WIN' | 'LOSS';
  ats_margin: number;
  confidence_tier: ConfidenceTier;
  bet_size: number;
  bet_recommendation: BetRecommendation;
}

export interface ResultsResponse {
  season: number;
  week: number;
  season_type: string;
  results: GameResult[];
  total_games: number;
  wins: number;
  losses: number;
  win_rate: number;
  status: 'profitable' | 'break_even' | 'review' | 'no_games' | 'no_lines';
}

// UI Helper Types
export type ConfidenceTier = Prediction['confidence_tier'];
export type BetRecommendation = Prediction['bet_recommendation'];

interface ColorStyle {
  bg: string;
  text: string;
  border?: string;
}

export const CONFIDENCE_COLORS: Record<ConfidenceTier, ColorStyle> = {
  'HIGH': { bg: 'bg-emerald-500', text: 'text-white' },
  'MEDIUM-HIGH': { bg: 'bg-emerald-400', text: 'text-black' },
  'MEDIUM': { bg: 'bg-amber-500', text: 'text-black' },
  'LOW': { bg: 'bg-amber-400', text: 'text-black' },
  'VERY LOW': { bg: 'bg-red-500', text: 'text-white' },
};

export const RECOMMENDATION_COLORS: Record<BetRecommendation, ColorStyle> = {
  'BET': { bg: 'bg-emerald-500', text: 'text-white', border: 'border-emerald-400' },
  'LEAN': { bg: 'bg-blue-500', text: 'text-white', border: 'border-blue-400' },
  'PASS': { bg: 'bg-slate-500', text: 'text-white', border: 'border-slate-400' },
};
