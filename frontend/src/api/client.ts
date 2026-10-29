import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Attach JWT access token if available, defaulting to guest-session for demo browsing
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("cinematch_token") || "guest-session";
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Intercept 401s to clear stale token
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      const isAuthUrl =
        error.config?.url?.includes("/auth/login") || error.config?.url?.includes("/auth/register");
      if (!isAuthUrl) {
        localStorage.removeItem("cinematch_token");
        window.dispatchEvent(new Event("cinematch_auth_change"));
      }
    }
    return Promise.reject(error);
  }
);
