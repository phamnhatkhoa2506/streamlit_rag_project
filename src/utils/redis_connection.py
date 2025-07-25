import os

from redis import Redis
from config import envConfig


def connect_redis(url: str) -> Redis:
    redis_client = Redis.from_url(url)
    redis_client.ping()

    return redis_client


redis_client = connect_redis(envConfig.REDIS_URL)
