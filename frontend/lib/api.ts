import type { PredictionsResponse, StatusResponse, AuthResponse, GamesResponse, ResultsResponse } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Log the API URL for debugging (visible in browser console)
if (typeof window !== 'undefined') {
  console.log('[API] Base URL:', API_BASE);
  if (API_BASE === 'http://localhost:8000') {
    console.warn('[API] WARNING: Using localhost - set NEXT_PUBLIC_API_URL in Vercel dashboard!');
  }
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
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
    } catch (error) {
      // Handle network-level errors (connection refused, DNS failure, etc.)
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        console.error('[API] Network error - cannot reach:', url);
        if (this.baseUrl === 'http://localhost:8000') {
          throw new Error('Backend not configured. Please set NEXT_PUBLIC_API_URL in Vercel dashboard.');
        }
        throw new Error(`Cannot connect to server. Please try again later.`);
      }
      throw error;
    }
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
    seasonType: string = 'regular'
  ): Promise<PredictionsResponse> {
    const params = new URLSearchParams({
      season: season.toString(),
      week: week.toString(),
      season_type: seasonType,
    });
    return this.fetch<PredictionsResponse>(`/api/predictions?${params}`);
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.fetch('/api/health');
  }

  async getResults(
    season: number,
    week: number,
    seasonType: string = 'regular'
  ): Promise<ResultsResponse> {
    const params = new URLSearchParams({
      season: season.toString(),
      week: week.toString(),
      season_type: seasonType,
    });
    return this.fetch<ResultsResponse>(`/api/results?${params}`);
  }
}

export const apiClient = new ApiClient();
export default apiClient;
