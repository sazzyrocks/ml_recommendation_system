export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
}

export interface Movie {
  id: number;
  ml_movie_id: number;
  tmdb_id?: number | null;
  title: string;
  year?: number | null;
  genres: string[];
  overview?: string | null;
  poster_path?: string | null;
}

export interface MovieDetail extends Movie {
  average_rating?: number | null;
  ratings_count: number;
}

export interface PaginatedMovies {
  items: Movie[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

export interface SimilarMovie {
  id: number;
  ml_movie_id: number;
  title: string;
  year?: number | null;
  genres: string[];
  similarity: number;
  poster_path?: string | null;
}

export interface SearchResult {
  id: number;
  ml_movie_id: number;
  title: string;
  year?: number | null;
  genres: string[];
  overview?: string | null;
  poster_path?: string | null;
  similarity: number;
}

export interface Rating {
  id: number;
  user_id: number;
  movie_id: number;
  rating: number;
  created_at: string;
  movie?: Movie | null;
}

export interface WatchlistEntry {
  id: number;
  user_id: number;
  movie_id: number;
  added_at: string;
  movie?: Movie | null;
}

export interface RecommendationItem {
  movie_id: number;
  title: string;
  year?: number | null;
  genres: string[];
  poster_path?: string | null;
  score: number;
  reason: string;
  top_source: string;
}

export interface RecommendationsResponse {
  items: RecommendationItem[];
  user_id: number;
  count: number;
}

export interface ModelBenchmark {
  model: string;
  "precision@k": number;
  "recall@k": number;
  "map@k": number;
  "ndcg@k": number;
  "hit_rate@k": number;
  catalog_coverage: number;
  avg_latency_ms: number;
  [key: string]: string | number;
}
