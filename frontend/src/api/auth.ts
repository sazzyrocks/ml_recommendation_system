import { apiClient } from "./client";
import type { AuthTokenResponse, User } from "./types";

export async function register(email: string, password: string): Promise<User> {
  const { data } = await apiClient.post<User>("/auth/register", { email, password });
  return data;
}

export async function login(email: string, password: string): Promise<AuthTokenResponse> {
  const { data } = await apiClient.post<AuthTokenResponse>("/auth/login", { email, password });
  return data;
}

export async function getMe(): Promise<User> {
  const { data } = await apiClient.get<User>("/auth/me");
  return data;
}
