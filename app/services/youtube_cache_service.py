import json
import hashlib
from typing import Optional, Dict, Any
from app.core.config import settings
from app.schemas.video import YouTubeSearchResponse
import redis.asyncio as redis
from datetime import timedelta

# from sentence_transformers import SentenceTransformer, util
# import numpy as np

class YouTubeCacheService:
    def __init__(self):
        self.redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=False
        )
        self.default_ttl = 3600  
        self.educational_ttl = 7200
        self.prefix = "youtube_search"

        # self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # self.vector_dim = 384
        # self.similarity_threshold = 0.8

    def _generate_cache_key(
        self,
        query: str,
        max_results: int,
        page_token: Optional[str],
        order: str,
        is_educational: bool = False
    ) -> str:
        params = {
            "q": query.lower().strip(),
            "max_results": max_results,
            "page_token": page_token or "",
            "order": order,
            "educational": is_educational,
        }
        param_string = json.dumps(params, sort_keys=True)
        hash_key = hashlib.md5(param_string.encode()).hexdigest()
        cache_type = "edu" if is_educational else "search"
        return f"{self.prefix}:{cache_type}:{hash_key}"

    def _generate_base_query_key(self, query: str, is_educational: bool = False) -> str:
        cache_type = "edu" if is_educational else "search"
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        return f"{self.prefix}:{cache_type}:base:{query_hash}"

    # async def _store_embedding(self, query: str, base_key: str, ttl: int):
    #     # embedding = self.model.encode(query, normalize_embeddings=True).astype(np.float32).tobytes()
    #     # meta_key = f"{base_key}:meta"
    #     # await self.redis_client.hset(meta_key, mapping={
    #     #     "query": query.lower().strip(),
    #     #     "embedding": embedding
    #     # })
    #     # await self.redis_client.expire(meta_key, ttl)
    #     return  # disabled

    # async def _find_similar_query(self, query: str, is_educational: bool = False) -> Optional[str]:
    #     # cache_type = "edu" if is_educational else "search"
    #     # meta_keys = await self.redis_client.keys(f"{self.prefix}:{cache_type}:base:*:meta")
    #     # if not meta_keys:
    #     #     return None

    #     # query_emb = self.model.encode(query, normalize_embeddings=True)
    #     # best_key = None
    #     # best_score = -1.0

    #     # for meta_key in meta_keys:
    #     #     data = await self.redis_client.hgetall(meta_key)
    #     #     if not data or b"embedding" not in data:
    #     #         continue

    #     #     emb_bytes = data[b"embedding"]
    #     #     emb = np.frombuffer(emb_bytes, dtype=np.float32)

    #     #     sim = util.cos_sim(query_emb, emb).item()
    #     #     if sim > best_score:
    #     #         best_score = sim
    #     #         best_key = meta_key

    #     # if best_key and best_score >= self.similarity_threshold:
    #     #     return best_key.decode().replace(":meta", "")
    #     return None  # disabled

    async def get_cached_search(
        self,
        query: str,
        max_results: int,
        page_token: Optional[str],
        order: str,
        is_educational: bool = False,
    ) -> Optional[YouTubeSearchResponse]:
        try:
            cache_key = self._generate_cache_key(query, max_results, page_token, order, is_educational)
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                data = json.loads(cached_data)
                return YouTubeSearchResponse(**data)

            # if not page_token:
            #     similar_base_key = await self._find_similar_query(query, is_educational)
            #     if similar_base_key:
            #         similar_cache_key = self._generate_cache_key(
            #             query, max_results, None, order, is_educational
            #         )
            #         cached_data = await self.redis_client.get(similar_cache_key)
            #         if cached_data:
            #             data = json.loads(cached_data)
            #             return YouTubeSearchResponse(**data)

            return None

        except Exception as e:
            print(f"Cache retrieval error: {e}")
            return None

    async def cache_search_results(
        self,
        query: str,
        max_results: int,
        page_token: Optional[str],
        order: str,
        results: YouTubeSearchResponse,
        is_educational: bool = False,
    ) -> None:
        try:
            cache_key = self._generate_cache_key(query, max_results, page_token, order, is_educational)
            cache_data = results.model_dump_json()
            ttl = self.educational_ttl if is_educational else self.default_ttl

            await self.redis_client.setex(cache_key, ttl, cache_data)

            # if not page_token:
            #     base_key = self._generate_base_query_key(query, is_educational)
            #     await self._store_embedding(query, base_key, ttl)

        except Exception as e:
            print(f"Cache storage error: {e}")

    async def invalidate_search_cache(self, pattern: Optional[str] = None) -> int:
        try:
            if pattern:
                search_pattern = f"{self.prefix}:{pattern}:*"
            else:
                search_pattern = f"{self.prefix}:*"

            keys = await self.redis_client.keys(search_pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0

        except Exception as e:
            print(f"Cache invalidation error: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        try:
            search_keys = await self.redis_client.keys(f"{self.prefix}:search:*")
            edu_keys = await self.redis_client.keys(f"{self.prefix}:edu:*")

            search_data_keys = [k for k in search_keys if not k.decode().endswith(':meta')]
            edu_data_keys = [k for k in edu_keys if not k.decode().endswith(':meta')]

            return {
                "total_cached_searches": len(search_data_keys) + len(edu_data_keys),
                "regular_searches": len(search_data_keys),
                "educational_searches": len(edu_data_keys),
                "cache_prefix": self.prefix,
            }

        except Exception as e:
            print(f"Cache stats error: {e}")
            return {"error": str(e)}

youtube_cache_service = YouTubeCacheService()
