"""
Redis / in-memory store provider
=================================
If REDIS_URL is set, connects to a real Redis server.
Otherwise, uses fakeredis (in-process), so no external service is needed
for single-container deployments (e.g. Railway).

Both the async API (FastAPI) and the sync Worker thread share the same
underlying FakeServer instance, so all state (jobs, sessions, queues) is
visible to both sides within the same process.
"""
import os

REDIS_URL = os.environ.get("REDIS_URL", "").strip()

if REDIS_URL:
    import redis as _redis
    import redis.asyncio as _aioredis

    # FIX #12: Use module-level connection pools so we don't create a new
    # connection per request. This enables proper connection reuse.
    _sync_pool = _redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
    _async_pool = _aioredis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)

    def get_sync_redis():
        return _redis.Redis(connection_pool=_sync_pool)

    async def get_async_redis():
        return _aioredis.Redis(connection_pool=_async_pool)

else:
    import fakeredis
    import fakeredis.aioredis as _fake_aioredis

    # One shared server so API and worker see the same data
    _server = fakeredis.FakeServer()

    def get_sync_redis():
        return fakeredis.FakeRedis(server=_server, decode_responses=True)

    async def get_async_redis():
        return _fake_aioredis.FakeRedis(server=_server, decode_responses=True)
