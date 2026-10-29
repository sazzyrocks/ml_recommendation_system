import { apiClient } from "./client";
import type { ModelBenchmark } from "./types";

export async function getModelBenchmarks(): Promise<ModelBenchmark[]> {
  const { data } = await apiClient.get<ModelBenchmark[]>("/analytics/models");
  return data;
}
