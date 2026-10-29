import React from "react";
import { useQuery } from "@tanstack/react-query";
import { getModelBenchmarks } from "../api/analytics";
import { EmptyState } from "../components/LoadingSkeleton";
import { PageHeader } from "../components/PageHeader";

const num = (v: unknown) => Number(v || 0);
const pct = (v: unknown) => `${(num(v) * 100).toFixed(1)}%`;

export const AnalyticsPage: React.FC = () => {
  const {
    data: models,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["analytics", "models"],
    queryFn: getModelBenchmarks,
  });

  if (isLoading) {
    return (
      <div className="mx-auto max-w-7xl animate-pulse space-y-6 px-4 py-10 sm:px-6 lg:px-8">
        <div className="h-8 w-64 rounded bg-cinema-850" />
        <div className="h-24 rounded-lg bg-cinema-850" />
        <div className="h-80 rounded-lg bg-cinema-850" />
      </div>
    );
  }

  if (error || !models || models.length === 0) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-16">
        <EmptyState
          title="No benchmark results"
          description="Run the offline evaluation so reports/results_test.csv exists, then reload."
        />
      </div>
    );
  }

  const ranked = [...models].sort((a, b) => num(b["ndcg@k"]) - num(a["ndcg@k"]));
  const best = ranked[0];
  const maxNdcg = num(best["ndcg@k"]) || 1;
  const hybrid = models.find((m) => m.model.toLowerCase().includes("hybrid"));

  const stats = [
    { label: "Best NDCG@10", value: num(best["ndcg@k"]).toFixed(4), note: best.model },
    {
      label: "Best hit rate@10",
      value: pct(Math.max(...models.map((m) => num(m["hit_rate@k"])))),
      note: "Users with at least one hit",
    },
    {
      label: "Hybrid latency",
      value: hybrid ? `${num(hybrid.avg_latency_ms).toFixed(2)} ms` : "—",
      note: "Mean per request",
    },
    {
      label: "Best catalog coverage",
      value: pct(Math.max(...models.map((m) => num(m.catalog_coverage)))),
      note: "Share of items ever recommended",
    },
  ];

  return (
    <div className="mx-auto max-w-7xl space-y-10 px-4 py-10 sm:px-6 lg:px-8">
      <PageHeader
        title="Model benchmarks"
        description="Offline results on the held-out temporal test split of MovieLens 100k, top-10 lists. Items seen in training are excluded."
      />

      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-cinema-800 bg-cinema-800 lg:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="bg-cinema-950 p-4">
            <dt className="text-xs text-cinema-500">{s.label}</dt>
            <dd className="mt-1 font-display text-2xl font-semibold tabular-nums text-white">
              {s.value}
            </dd>
            <dd className="mt-0.5 truncate text-xs text-cinema-500">{s.note}</dd>
          </div>
        ))}
      </dl>

      <section className="space-y-4">
        <h2 className="font-display text-lg font-semibold text-white">NDCG@10 by model</h2>
        <div className="space-y-3">
          {ranked.map((row) => {
            const value = num(row["ndcg@k"]);
            const isHybrid = row === hybrid;
            return (
              <div
                key={row.model}
                className="grid grid-cols-[10rem_1fr_4rem] items-center gap-3 text-sm"
              >
                <span className={`truncate ${isHybrid ? "text-crimson-400" : "text-cinema-300"}`}>
                  {row.model}
                </span>
                <div className="h-2 overflow-hidden rounded-full bg-cinema-850">
                  <div
                    className={`h-full rounded-full ${
                      isHybrid ? "bg-gradient-to-r from-crimson-500 to-amber-400" : "bg-cinema-600"
                    }`}
                    style={{ width: `${Math.max(2, (value / maxNdcg) * 100)}%` }}
                  />
                </div>
                <span className="text-right font-mono text-xs text-cinema-400">
                  {value.toFixed(4)}
                </span>
              </div>
            );
          })}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="font-display text-lg font-semibold text-white">All metrics</h2>
        <div className="overflow-x-auto rounded-lg border border-cinema-800">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-cinema-800 text-cinema-500">
              <tr>
                {[
                  "Model",
                  "P@10",
                  "R@10",
                  "MAP@10",
                  "NDCG@10",
                  "Hit@10",
                  "Coverage",
                  "Latency (ms)",
                ].map((h, i) => (
                  <th key={h} className={`px-4 py-3 font-medium ${i > 0 ? "text-right" : ""}`}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-cinema-800/70 font-mono text-cinema-300">
              {models.map((row) => (
                <tr
                  key={row.model}
                  className={
                    row === hybrid ? "bg-crimson-500/10 text-white" : "hover:bg-cinema-900"
                  }
                >
                  <td className="px-4 py-3 font-sans font-medium">{row.model}</td>
                  <td className="px-4 py-3 text-right">{num(row["precision@k"]).toFixed(4)}</td>
                  <td className="px-4 py-3 text-right">{num(row["recall@k"]).toFixed(4)}</td>
                  <td className="px-4 py-3 text-right">{num(row["map@k"]).toFixed(4)}</td>
                  <td className="px-4 py-3 text-right text-white">
                    {num(row["ndcg@k"]).toFixed(4)}
                  </td>
                  <td className="px-4 py-3 text-right">{pct(row["hit_rate@k"])}</td>
                  <td className="px-4 py-3 text-right">{pct(row.catalog_coverage)}</td>
                  <td className="px-4 py-3 text-right text-cinema-400">
                    {num(row.avg_latency_ms).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
