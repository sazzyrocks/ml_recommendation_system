import { apiClient } from "./client";
import type { WatchlistEntry } from "./types";

export async function getWatchlist(): Promise<WatchlistEntry[]> {
  const { data } = await apiClient.get<WatchlistEntry[]>("/watchlist");
  return data;
}

export async function addToWatchlist(movieId: number): Promise<WatchlistEntry> {
  const { data } = await apiClient.post<WatchlistEntry>("/watchlist", {
    movie_id: movieId,
  });
  return data;
}

export async function removeFromWatchlist(movieId: number): Promise<void> {
  await apiClient.delete(`/watchlist/${movieId}`);
}
