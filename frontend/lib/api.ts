import type { PredictionsResponse, StatusResponse, AuthResponse, GamesResponse } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async authenticate(password: string): Promise<AuthResponse> {
    return this.fetch<AuthResponse>('/api/auth', {
      method: 'POST',
      body: JSON.stringify({ password }),
    });
  }

  async getStatus(): Promise<StatusResponse> {
    return this.fetch<StatusResponse>('/api/status');
  }

  async getGames(season: number, week: number, seasonType: string = 'regular'): Promise<GamesResponse> {
    const params = new URLSearchParams({
      season: season.toString(),
      week: week.toString(),
      season_type: seasonType,
    });
    return this.fetch<GamesResponse>(`/api/games?${params}`);
  }

  async getPredictions(
    season: number,
    week: number,
    seasonType: string = 'regular',
    bankroll: number = 1000
  ): Promise<PredictionsResponse> {
    const params = new URLSearchParams({
      season: season.toString(),
      week: week.toString(),
      season_type: seasonType,
      bankroll: bankroll.toString(),
    });
    return this.fetch<PredictionsResponse>(`/api/predictions?${params}`);
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.fetch('/api/health');
  }
}

export const api = new ApiClient();
export default api;
