import { apiClient } from "./client";
import type { MovieDetail, PaginatedMovies, SearchResult, SimilarMovie } from "./types";

export interface MovieQueryParams {
  page?: number;
  limit?: number;
  genre?: string;
  year?: number;
  year_min?: number;
  year_max?: number;
  sort?: "id" | "popular";
}

export async function getMovies(params: MovieQueryParams = {}): Promise<PaginatedMovies> {
  const { data } = await apiClient.get<PaginatedMovies>("/movies", { params });
  return data;
}

export async function getMovie(id: number): Promise<MovieDetail> {
  const { data } = await apiClient.get<MovieDetail>(`/movies/${id}`);
  return data;
}

export async function getSimilarMovies(id: number, k: number = 10): Promise<SimilarMovie[]> {
  const { data } = await apiClient.get<SimilarMovie[]>(`/movies/${id}/similar`, {
    params: { k },
  });
  return data;
}

export interface SearchQueryParams {
  q: string;
  k?: number;
  genre?: string;
  year_min?: number;
  year_max?: number;
}

export async function searchMovies(params: SearchQueryParams): Promise<SearchResult[]> {
  const { data } = await apiClient.get<SearchResult[]>("/search", { params });
  return data;
}
