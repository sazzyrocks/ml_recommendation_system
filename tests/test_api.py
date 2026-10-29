"""
tests/test_api.py
-----------------
Integration tests for CineMatch AI FastAPI backend.
Tests run against a separate isolated SQLite database using httpx.AsyncClient.

Covers:
- /health probe
- Auth flow: register, duplicate rejection, login, bad password rejection, /auth/me
- Anonymous user rejection on protected routes (401)
- Movie catalog endpoints: /movies, /movies/{id}, /movies/{id}/similar, /search?q=
- Ratings CRUD: POST, PUT, DELETE, GET /ratings/me
- User isolation: User B cannot modify or delete User A's ratings/watchlist
- Watchlist: GET, POST, DELETE
- Recommendations: GET /recommendations/me (with fold-in reasoning)
- Analytics: GET /analytics/models
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.database import Base, get_db
from backend.main import app
from backend.models.entities import Movie

# Dedicated test database file
TEST_DB_PATH = "./test_cinematch_suite.db"
TEST_DATABASE_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@pytest_asyncio.fixture(scope="function", autouse=True)
async def prepare_test_db():
    """Create schema and seed sample catalog movies once for the test suite."""
    app.dependency_overrides[get_db] = override_get_db

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Seed 3 catalog movies
    async with TestSessionLocal() as session:
        movies = [
            Movie(
                ml_movie_id=79132,
                tmdb_id=27205,
                title="Inception (2010)",
                year=2010,
                genres=["Action", "Adventure", "Sci-Fi"],
                overview="A thief who steals corporate secrets through the use of dream-sharing technology.",
                poster_path="/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
                embedding=[0.01] * 384,
            ),
            Movie(
                ml_movie_id=1,
                tmdb_id=862,
                title="Toy Story (1995)",
                year=1995,
                genres=["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
                overview="A little boy named Andy loves to be in his room, playing with his toys.",
                poster_path="/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
                embedding=[-0.01] * 384,
            ),
            Movie(
                ml_movie_id=2571,
                tmdb_id=603,
                title="The Matrix (1999)",
                year=1999,
                genres=["Action", "Sci-Fi"],
                overview="A computer hacker learns about the true nature of reality and his role in the war.",
                poster_path="/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
                embedding=[0.02] * 384,
            ),
        ]
        session.add_all(movies)
        await session.commit()

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client bound directly to the FastAPI app via ASGITransport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ── 1. Health Probe ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data


# ── 2. Auth Flow & Anonymous Route Protection ────────────────────────────────


@pytest.mark.asyncio
async def test_auth_registration_and_login(client: AsyncClient):
    # Anonymous request to /auth/me should fail with 401
    anon_resp = await client.get("/auth/me")
    assert anon_resp.status_code == 401

    # Register User A
    reg_resp = await client.post(
        "/auth/register",
        json={"email": "alice@cinematch.ai", "password": "securepassword123"},
    )
    assert reg_resp.status_code == 201
    user_a = reg_resp.json()
    assert user_a["email"] == "alice@cinematch.ai"
    assert "id" in user_a

    # Duplicate registration should return 400
    dup_resp = await client.post(
        "/auth/register",
        json={"email": "alice@cinematch.ai", "password": "anotherpassword123"},
    )
    assert dup_resp.status_code == 400

    # Login with wrong password should fail with 401
    bad_login = await client.post(
        "/auth/login",
        json={"email": "alice@cinematch.ai", "password": "wrongpassword"},
    )
    assert bad_login.status_code == 401

    # Login with correct password
    login_resp = await client.post(
        "/auth/login",
        json={"email": "alice@cinematch.ai", "password": "securepassword123"},
    )
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # Verify /auth/me with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    me_resp = await client.get("/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "alice@cinematch.ai"


# ── 3. Movie Catalog & Search ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_movie_catalog_and_search(client: AsyncClient):
    # GET /movies pagination
    res = await client.get("/movies?page=1&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert len(data["items"]) >= 3
    assert data["total"] >= 3

    first_movie = data["items"][0]
    movie_id = first_movie["id"]

    # GET /movies/{id}
    detail_res = await client.get(f"/movies/{movie_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["title"] == first_movie["title"]

    # Non-existent movie
    missing_res = await client.get("/movies/999999")
    assert missing_res.status_code == 404

    # GET /movies/{id}/similar
    sim_res = await client.get(f"/movies/{movie_id}/similar?k=5")
    assert sim_res.status_code == 200
    assert isinstance(sim_res.json(), list)

    # GET /search?q=
    search_res = await client.get("/search?q=dream")
    assert search_res.status_code == 200
    search_hits = search_res.json()
    assert isinstance(search_hits, list)


# ── 4. Ratings CRUD & User Data Isolation ─────────────────────────────────────


@pytest.mark.asyncio
async def test_ratings_crud_and_isolation(client: AsyncClient):
    # Register Alice (User A) and Bob (User B)
    await client.post(
        "/auth/register",
        json={"email": "rating_alice@cinematch.ai", "password": "Password123!"},
    )
    alice_login = await client.post(
        "/auth/login",
        json={"email": "rating_alice@cinematch.ai", "password": "Password123!"},
    )
    alice_token = alice_login.json()["access_token"]
    alice_headers = {"Authorization": f"Bearer {alice_token}"}

    await client.post(
        "/auth/register",
        json={"email": "rating_bob@cinematch.ai", "password": "Password123!"},
    )
    bob_login = await client.post(
        "/auth/login",
        json={"email": "rating_bob@cinematch.ai", "password": "Password123!"},
    )
    bob_token = bob_login.json()["access_token"]
    bob_headers = {"Authorization": f"Bearer {bob_token}"}

    # Get movie ID 1
    movies_res = await client.get("/movies?limit=1")
    movie_id = movies_res.json()["items"][0]["id"]

    # 1. Anonymous rating should fail (401)
    anon_post = await client.post("/ratings", json={"movie_id": movie_id, "rating": 4.5})
    assert anon_post.status_code == 401

    # 2. Alice rates movie (POST /ratings)
    r1 = await client.post(
        "/ratings",
        json={"movie_id": movie_id, "rating": 4.5},
        headers=alice_headers,
    )
    assert r1.status_code == 201
    assert r1.json()["rating"] == 4.5

    # 3. Duplicate rating via POST should be rejected (400)
    r_dup = await client.post(
        "/ratings",
        json={"movie_id": movie_id, "rating": 5.0},
        headers=alice_headers,
    )
    assert r_dup.status_code == 400

    # 4. Rating out of bounds (<0.5 or >5.0) rejected by Pydantic validation (422)
    r_invalid = await client.post(
        "/ratings",
        json={"movie_id": movie_id, "rating": 6.0},
        headers=alice_headers,
    )
    assert r_invalid.status_code == 422

    # 5. Alice updates rating via PUT /ratings/{movie_id}
    r_update = await client.put(
        f"/ratings/{movie_id}",
        json={"rating": 5.0},
        headers=alice_headers,
    )
    assert r_update.status_code == 200
    assert r_update.json()["rating"] == 5.0

    # 6. Alice lists own ratings (GET /ratings/me)
    alice_ratings = await client.get("/ratings/me", headers=alice_headers)
    assert alice_ratings.status_code == 200
    items = alice_ratings.json()
    assert len(items) == 1
    assert items[0]["rating"] == 5.0

    # 7. ISOLATION: Bob cannot see, update, or delete Alice's rating
    bob_ratings = await client.get("/ratings/me", headers=bob_headers)
    assert bob_ratings.status_code == 200
    assert len(bob_ratings.json()) == 0  # Bob sees none of Alice's ratings

    bob_update_attempt = await client.put(
        f"/ratings/{movie_id}",
        json={"rating": 1.0},
        headers=bob_headers,
    )
    assert bob_update_attempt.status_code == 404  # Bob has no rating on movie_id to update

    bob_delete_attempt = await client.delete(
        f"/ratings/{movie_id}",
        headers=bob_headers,
    )
    assert bob_delete_attempt.status_code == 404  # Bob cannot delete Alice's rating

    # Alice's rating is still intact
    alice_check = await client.get("/ratings/me", headers=alice_headers)
    assert len(alice_check.json()) == 1

    # 8. Alice deletes own rating (DELETE /ratings/{movie_id})
    del_resp = await client.delete(f"/ratings/{movie_id}", headers=alice_headers)
    assert del_resp.status_code == 204

    # Alice now has 0 ratings
    alice_empty = await client.get("/ratings/me", headers=alice_headers)
    assert len(alice_empty.json()) == 0


# ── 5. Watchlist CRUD & User Data Isolation ───────────────────────────────────


@pytest.mark.asyncio
async def test_watchlist_crud_and_isolation(client: AsyncClient):
    # Register Charlie and Dave
    await client.post(
        "/auth/register",
        json={"email": "watch_charlie@cinematch.ai", "password": "Password123!"},
    )
    charlie_login = await client.post(
        "/auth/login",
        json={"email": "watch_charlie@cinematch.ai", "password": "Password123!"},
    )
    charlie_token = charlie_login.json()["access_token"]
    charlie_headers = {"Authorization": f"Bearer {charlie_token}"}

    await client.post(
        "/auth/register",
        json={"email": "watch_dave@cinematch.ai", "password": "Password123!"},
    )
    dave_login = await client.post(
        "/auth/login",
        json={"email": "watch_dave@cinematch.ai", "password": "Password123!"},
    )
    dave_token = dave_login.json()["access_token"]
    dave_headers = {"Authorization": f"Bearer {dave_token}"}

    movies_res = await client.get("/movies?limit=1")
    movie_id = movies_res.json()["items"][0]["id"]

    # Anonymous user rejected (401)
    anon_watch = await client.get("/watchlist")
    assert anon_watch.status_code == 401

    # Charlie adds movie to watchlist
    w1 = await client.post(
        "/watchlist",
        json={"movie_id": movie_id},
        headers=charlie_headers,
    )
    assert w1.status_code == 201
    assert w1.json()["movie_id"] == movie_id

    # Charlie tries duplicate add -> 400
    w_dup = await client.post(
        "/watchlist",
        json={"movie_id": movie_id},
        headers=charlie_headers,
    )
    assert w_dup.status_code == 400

    # Charlie lists watchlist -> 1 item
    charlie_list = await client.get("/watchlist", headers=charlie_headers)
    assert charlie_list.status_code == 200
    assert len(charlie_list.json()) == 1

    # ISOLATION: Dave sees empty watchlist
    dave_list = await client.get("/watchlist", headers=dave_headers)
    assert dave_list.status_code == 200
    assert len(dave_list.json()) == 0

    # Dave tries to delete Charlie's watchlist entry -> 404
    dave_del = await client.delete(f"/watchlist/{movie_id}", headers=dave_headers)
    assert dave_del.status_code == 404

    # Charlie deletes from watchlist -> 204
    charlie_del = await client.delete(f"/watchlist/{movie_id}", headers=charlie_headers)
    assert charlie_del.status_code == 204

    charlie_empty = await client.get("/watchlist", headers=charlie_headers)
    assert len(charlie_empty.json()) == 0


# ── 6. Recommendations & Personalization ──────────────────────────────────────


@pytest.mark.asyncio
async def test_recommendations_endpoint(client: AsyncClient):
    # Anonymous rejected
    anon_rec = await client.get("/recommendations/me")
    assert anon_rec.status_code == 401

    # Authenticated user
    await client.post(
        "/auth/register",
        json={"email": "recs_user@cinematch.ai", "password": "Password123!"},
    )
    login_resp = await client.post(
        "/auth/login",
        json={"email": "recs_user@cinematch.ai", "password": "Password123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

    # Add a rating to personalize fold-in
    movies_res = await client.get("/movies?limit=1")
    movie_id = movies_res.json()["items"][0]["id"]
    await client.post(
        "/ratings",
        json={"movie_id": movie_id, "rating": 5.0},
        headers=headers,
    )

    rec_res = await client.get("/recommendations/me?k=5", headers=headers)
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert "items" in rec_data
    assert "user_id" in rec_data
    assert isinstance(rec_data["items"], list)
    if rec_data["items"]:
        first_rec = rec_data["items"][0]
        assert "score" in first_rec
        assert "reason" in first_rec
        assert "top_source" in first_rec


# ── 7. Analytics & Meta Endpoints ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analytics_models_endpoint(client: AsyncClient):
    res = await client.get("/analytics/models")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "model" in data[0]


# ── 8. Hardening & Edge Cases (Phase 12) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_recommendations_new_user_cold_start(client: AsyncClient):
    """Test recommendation generation for a brand-new user with 0 ratings (cold-start)."""
    reg = await client.post(
        "/auth/register",
        json={"email": "coldstart_user@cinematch.ai", "password": "Password123!"},
    )
    login_resp = await client.post(
        "/auth/login",
        json={"email": "coldstart_user@cinematch.ai", "password": "Password123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

    # Should not crash; gracefully returns popularity / general items
    res = await client.get("/recommendations/me?k=5", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    assert data["user_id"] == reg.json()["id"]


@pytest.mark.asyncio
async def test_user_who_rated_catalog_movies(client: AsyncClient):
    """Ensure recommended items strictly exclude items already rated by the user."""
    await client.post(
        "/auth/register",
        json={"email": "heavy_rater@cinematch.ai", "password": "Password123!"},
    )
    login_resp = await client.post(
        "/auth/login",
        json={"email": "heavy_rater@cinematch.ai", "password": "Password123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

    # Fetch all current movies in test DB
    m_res = await client.get("/movies")
    movies = m_res.json()["items"]
    assert len(movies) >= 2

    # Rate the first movie
    first_id = movies[0]["id"]
    await client.post(
        "/ratings",
        json={"movie_id": first_id, "rating": 5.0},
        headers=headers,
    )

    rec_res = await client.get("/recommendations/me?k=10", headers=headers)
    assert rec_res.status_code == 200
    recommended_ids = [item["movie_id"] for item in rec_res.json()["items"]]
    assert (
        first_id not in recommended_ids
    ), "Recommendations must not include movies user already rated"


@pytest.mark.asyncio
async def test_unknown_movie_id_edge_cases(client: AsyncClient):
    """Verify that operations on non-existent movie IDs cleanly return 404."""
    await client.post(
        "/auth/register",
        json={"email": "edge_tester@cinematch.ai", "password": "Password123!"},
    )
    login_resp = await client.post(
        "/auth/login",
        json={"email": "edge_tester@cinematch.ai", "password": "Password123!"},
    )
    headers = {"Authorization": f"Bearer {login_resp.json()['access_token']}"}

    fake_id = 999999

    # Movie details
    assert (await client.get(f"/movies/{fake_id}")).status_code == 404

    # Similar movies
    assert (await client.get(f"/movies/{fake_id}/similar")).status_code == 404

    # Create rating for non-existent movie
    r_post = await client.post(
        "/ratings", json={"movie_id": fake_id, "rating": 4.0}, headers=headers
    )
    assert r_post.status_code == 404

    # Update rating for non-existent movie
    r_put = await client.put(f"/ratings/{fake_id}", json={"rating": 3.0}, headers=headers)
    assert r_put.status_code == 404

    # Delete rating for non-existent movie
    r_del = await client.delete(f"/ratings/{fake_id}", headers=headers)
    assert r_del.status_code == 404

    # Add non-existent movie to watchlist
    w_post = await client.post("/watchlist", json={"movie_id": fake_id}, headers=headers)
    assert w_post.status_code == 404

    # Delete non-existent movie from watchlist
    w_del = await client.delete(f"/watchlist/{fake_id}", headers=headers)
    assert w_del.status_code == 404


@pytest.mark.asyncio
async def test_password_rules_validation(client: AsyncClient):
    """Verify strong password validation rules."""
    # Password too short (<8 chars)
    short_resp = await client.post(
        "/auth/register",
        json={"email": "short_pw@cinematch.ai", "password": "abc1"},
    )
    assert short_resp.status_code == 422

    # Password without digits
    no_num_resp = await client.post(
        "/auth/register",
        json={"email": "no_num@cinematch.ai", "password": "passwordonlyletters"},
    )
    assert no_num_resp.status_code == 422


@pytest.mark.asyncio
async def test_model_backed_paths_and_genre_filter(client: AsyncClient, monkeypatch):
    """Exercise the ML-backed branches (normally skipped because lifespan doesn't run in tests)."""
    from backend.ml_service import ml_service

    hit = {
        "movieId": 1,
        "item_idx": 0,
        "title": "Toy Story",
        "year": 1995,
        "genres": ["Animation"],
        "similarity": 0.9,
        "similarity_score": 0.9,
        "score": 0.9,
        "reason": "Because you rated Inception",
        "top_source": "implicit_als",
    }
    fake_search = type("FakeSearch", (), {"search": lambda self, query, k, filters: [hit]})()
    monkeypatch.setattr(ml_service, "find_similar_movies", lambda mid, k=10: [hit])
    monkeypatch.setattr(ml_service, "semantic_search", fake_search)
    monkeypatch.setattr(ml_service, "recommend_for_user", lambda user_ratings_db, k=20: [hit])
    monkeypatch.setattr(ml_service, "popular_mids", [1, 79132])

    # Genre filter on JSON column
    res = await client.get("/api/v1/movies", params={"genre": "Animation"})
    assert res.status_code == 200
    assert [m["ml_movie_id"] for m in res.json()["items"]] == [1]

    # Popularity sort follows MLService ordering
    res = await client.get("/api/v1/movies", params={"sort": "popular"})
    assert res.status_code == 200
    assert [m["ml_movie_id"] for m in res.json()["items"]] == [1, 79132]

    inception_id = (await client.get("/api/v1/movies", params={"genre": "Sci-Fi"})).json()["items"][
        0
    ]["id"]

    res = await client.get(f"/api/v1/movies/{inception_id}/similar")
    assert res.status_code == 200
    assert res.json()[0]["ml_movie_id"] == 1
    assert res.json()[0]["similarity"] == 0.9

    res = await client.get("/api/v1/search", params={"q": "toys"})
    assert res.status_code == 200
    assert res.json()[0]["ml_movie_id"] == 1

    await client.post(
        "/api/v1/auth/register", json={"email": "ml@example.com", "password": "password123"}
    )
    login = await client.post(
        "/api/v1/auth/login", json={"email": "ml@example.com", "password": "password123"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    res = await client.get("/api/v1/recommendations/me", headers=headers)
    assert res.status_code == 200
    assert res.json()["items"][0]["title"] == "Toy Story (1995)"
