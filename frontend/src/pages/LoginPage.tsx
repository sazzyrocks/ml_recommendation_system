import React, { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Film, Lock, Mail, ArrowRight, AlertCircle, KeyRound } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export const LoginPage: React.FC = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || "/dashboard";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await login(email, password);
      navigate(from, { replace: true });
    } catch (err: any) {
      setError(err.response?.data?.detail || "Invalid email or password. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFillDemo = () => {
    setEmail("demo@cinematch.ai");
    setPassword("Password123!");
    setError(null);
  };

  return (
    <div className="min-h-[75vh] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md space-y-6">
        {/* Brand header */}
        <div className="text-center space-y-2">
          <div className="inline-flex w-12 h-12 rounded-2xl bg-crimson-500 items-center justify-center shadow-lg shadow-crimson-500/30">
            <Film className="w-6 h-6 text-white" />
          </div>
          <h2 className="font-display font-extrabold text-2xl sm:text-3xl text-white tracking-tight">
            Welcome back to CineMatch
          </h2>
          <p className="text-xs sm:text-sm text-cinema-400">
            Sign in to access your adaptive recommendations and watchlist
          </p>
        </div>

        {/* Card */}
        <div className="p-8 rounded-3xl bg-cinema-900/90 border border-cinema-750/80 shadow-2xl space-y-6 glass-panel">
          {error && (
            <div className="p-3.5 rounded-xl bg-crimson-500/10 border border-crimson-500/30 flex items-start space-x-2 text-xs text-crimson-400">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-cinema-300">Email Address</label>
              <div className="relative">
                <Mail className="w-4 h-4 text-cinema-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@example.com"
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-cinema-850 border border-cinema-700 text-sm text-white placeholder-cinema-500 focus:outline-none focus:border-crimson-500 focus:ring-1 focus:ring-crimson-500 transition-all"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-cinema-300">Password</label>
              <div className="relative">
                <Lock className="w-4 h-4 text-cinema-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-cinema-850 border border-cinema-700 text-sm text-white placeholder-cinema-500 focus:outline-none focus:border-crimson-500 focus:ring-1 focus:ring-crimson-500 transition-all"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full py-3 rounded-xl font-semibold text-sm text-white bg-crimson-500 hover:bg-crimson-600 disabled:opacity-50 transition-all shadow-lg shadow-crimson-500/25 flex items-center justify-center space-x-2 cursor-pointer"
            >
              <span>{isSubmitting ? "Signing in..." : "Sign In"}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          {/* Quick Demo Login Option */}
          <div className="pt-2 border-t border-cinema-800">
            <button
              type="button"
              onClick={handleFillDemo}
              className="w-full py-2.5 px-3 rounded-xl text-xs font-medium text-cinema-300 hover:text-white bg-cinema-800/80 hover:bg-cinema-750 border border-cinema-700 flex items-center justify-center space-x-2 transition-colors cursor-pointer"
            >
              <KeyRound className="w-3.5 h-3.5 text-gold-400" />
              <span>Fill 1-Click Demo Account (demo@cinematch.ai)</span>
            </button>
          </div>
        </div>

        <p className="text-center text-xs text-cinema-400">
          Don't have an account?{" "}
          <Link to="/register" className="text-crimson-400 font-semibold hover:underline">
            Register now
          </Link>
        </p>
      </div>
    </div>
  );
};
