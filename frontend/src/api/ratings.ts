import { apiClient } from "./client";
import type { Rating } from "./types";

export async function createRating(movieId: number, rating: number): Promise<Rating> {
  const { data } = await apiClient.post<Rating>("/ratings", {
    movie_id: movieId,
    rating,
  });
  return data;
}

export async function updateRating(movieId: number, rating: number): Promise<Rating> {
  const { data } = await apiClient.put<Rating>(`/ratings/${movieId}`, {
    rating,
  });
  return data;
}

export async function deleteRating(movieId: number): Promise<void> {
  await apiClient.delete(`/ratings/${movieId}`);
}

export async function getMyRatings(): Promise<Rating[]> {
  const { data } = await apiClient.get<Rating[]>("/ratings/me");
  return data;
}
