import { apiClient } from "./client";
import type { RecommendationsResponse } from "./types";

export async function getMyRecommendations(k: number = 20): Promise<RecommendationsResponse> {
  const { data } = await apiClient.get<RecommendationsResponse>("/recommendations/me", {
    params: { k },
  });
  return data;
}
