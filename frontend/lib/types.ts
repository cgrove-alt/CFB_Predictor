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

// UI Helper Types
export type ConfidenceTier = Prediction['confidence_tier'];
export type BetRecommendation = Prediction['bet_recommendation'];

export const CONFIDENCE_COLORS: Record<ConfidenceTier, string> = {
  'HIGH': 'bg-emerald-500 text-white',
  'MEDIUM-HIGH': 'bg-emerald-400 text-black',
  'MEDIUM': 'bg-amber-500 text-black',
  'LOW': 'bg-amber-400 text-black',
  'VERY LOW': 'bg-red-500 text-white',
};

export const RECOMMENDATION_COLORS: Record<BetRecommendation, string> = {
  'BET': 'bg-emerald-500 text-white',
  'LEAN': 'bg-blue-500 text-white',
  'PASS': 'bg-slate-500 text-white',
};
