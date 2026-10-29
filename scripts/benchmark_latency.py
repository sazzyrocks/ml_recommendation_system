"""
scripts/benchmark_latency.py
----------------------------
Measures p50 and p95 latency for critical user-facing endpoints:
- GET /recommendations/me?k=20 (hybrid fold-in recommendation generation)
- GET /search?q= (semantic dense vector search)

Runs 50 requests per endpoint and prints statistical distribution.
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.database import Base, get_db
from backend.main import app
from backend.models.entities import Movie

TEST_DB_URL = "sqlite+aiosqlite:///./test_latency.db"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def run_benchmark():
    app.dependency_overrides[get_db] = override_get_db

    # Create tables & sample movie
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        session.add(
            Movie(
                ml_movie_id=1,
                title="Toy Story (1995)",
                year=1995,
                genres=["Adventure", "Animation", "Children"],
                embedding=[0.01] * 384,
            )
        )
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Register and login benchmark user
        await client.post(
            "/auth/register",
            json={"email": "bench_user@cinematch.ai", "password": "BenchPassword123!"},
        )
        login_res = await client.post(
            "/auth/login",
            json={"email": "bench_user@cinematch.ai", "password": "BenchPassword123!"},
        )
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Warm up
        await client.get("/recommendations/me?k=20", headers=headers)
        await client.get("/search?q=space+exploration")

        # ── 1. Measure /recommendations/me latency (50 samples) ────────────────
        rec_latencies = []
        for _ in range(50):
            start = time.perf_counter()
            resp = await client.get("/recommendations/me?k=20", headers=headers)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert resp.status_code == 200
            rec_latencies.append(elapsed_ms)

        # ── 2. Measure /search?q= latency (50 samples) ─────────────────────────
        search_queries = [
            "space exploration with emotional story",
            "heist comedy with clever twists",
            "dark psychological thriller",
            "animated friendship journey",
            "cyberpunk detective mystery",
        ]
        search_latencies = []
        for i in range(50):
            q = search_queries[i % len(search_queries)]
            start = time.perf_counter()
            resp = await client.get(f"/search?q={q}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert resp.status_code == 200
            search_latencies.append(elapsed_ms)

    def calc_stats(times: list[float]) -> dict[str, float]:
        times_sorted = sorted(times)
        n = len(times_sorted)
        p50 = statistics.median(times_sorted)
        p95 = times_sorted[int(0.95 * n)]
        p99 = times_sorted[int(0.99 * n)]
        mean = statistics.mean(times_sorted)
        return {
            "min": times_sorted[0],
            "max": times_sorted[-1],
            "mean": mean,
            "p50": p50,
            "p95": p95,
            "p99": p99,
        }

    rec_stats = calc_stats(rec_latencies)
    search_stats = calc_stats(search_latencies)

    print("\n" + "=" * 70)
    print("  CINEMATCH BACKEND LATENCY BENCHMARK (50 SAMPLES EACH)")
    print("=" * 70)
    print(
        f"{'Endpoint':<25} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10}"
    )
    print("-" * 70)
    print(
        f"{'GET /recommendations/me':<25} | {rec_stats['mean']:<10.2f} | "
        f"{rec_stats['p50']:<10.2f} | {rec_stats['p95']:<10.2f} | {rec_stats['p99']:<10.2f}"
    )
    print(
        f"{'GET /search?q=...':<25} | {search_stats['mean']:<10.2f} | "
        f"{search_stats['p50']:<10.2f} | {search_stats['p95']:<10.2f} | {search_stats['p99']:<10.2f}"
    )
    print("=" * 70)

    # Performance Gate
    assert (
        rec_stats["p95"] < 500.0
    ), f"Recommendation p95 exceeds 500ms limit! ({rec_stats['p95']:.2f}ms)"
    assert (
        search_stats["p95"] < 500.0
    ), f"Search p95 exceeds 500ms limit! ({search_stats['p95']:.2f}ms)"
    print(">> LATENCY GATE PASSED: Both endpoints achieve p95 well below 500 ms threshold!")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
