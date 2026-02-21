from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)


class QdrantMenuVectorStore:
    """
    Qdrant-backed menu vector index for AI/v2.

    Design rules:
    - RDB remains source of truth.
    - Vector index is used for candidate retrieval only.
    """

    def __init__(
        self,
        openai_client: Any,
        collection_name: str = "kiosk_menu_v2",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> None:
        self.openai_client = openai_client
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.vector_size = int(vector_size or os.getenv("QDRANT_VECTOR_SIZE", "1536"))
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=False,
        )
        self._ready = False

    async def ensure_collection(self) -> None:
        await asyncio.to_thread(self._ensure_collection_sync)
        self._ready = True

    def _ensure_collection_sync(self) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(size=self.vector_size, distance=qm.Distance.COSINE),
                on_disk_payload=True,
            )

        payload_indexes = [
            ("menuItemId", qm.PayloadSchemaType.KEYWORD),
            ("name", qm.PayloadSchemaType.TEXT),
            ("categoryId", qm.PayloadSchemaType.KEYWORD),
            ("allergies", qm.PayloadSchemaType.KEYWORD),
            ("ingredients", qm.PayloadSchemaType.KEYWORD),
            ("tags", qm.PayloadSchemaType.KEYWORD),
            ("price", qm.PayloadSchemaType.FLOAT),
            ("isSetLike", qm.PayloadSchemaType.BOOL),
        ]
        for field_name, schema in payload_indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                # index may already exist
                pass

    async def upsert_menu_items(
        self,
        menu_items: List[Dict[str, Any]],
        detail_provider: Optional[Callable[[str], Awaitable[Optional[Dict[str, Any]]]]] = None,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        if not self._ready:
            await self.ensure_collection()

        normalized: List[Dict[str, Any]] = []
        for it in menu_items:
            mid = str(it.get("menuItemId") or "").strip()
            name = str(it.get("name") or "").strip()
            if not mid or not name:
                continue

            detail = None
            if detail_provider:
                try:
                    detail = await detail_provider(mid)
                except Exception:
                    detail = None

            merged = dict(it)
            if isinstance(detail, dict):
                merged.update(detail)
            normalized.append(merged)

        texts = [self._build_embedding_text(it) for it in normalized]
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            part = await asyncio.to_thread(self._embed_sync, batch)
            embeddings.extend(part)

        points: List[qm.PointStruct] = []
        for item, vec in zip(normalized, embeddings):
            payload = self._build_payload(item)
            points.append(
                qm.PointStruct(
                    id=str(payload["menuItemId"]),
                    vector=vec,
                    payload=payload,
                )
            )

        if points:
            await asyncio.to_thread(
                self.client.upsert,
                self.collection_name,
                points,
            )

        return {
            "collection": self.collection_name,
            "upserted": len(points),
        }

    async def search_menus(
        self,
        query: str,
        top_k: int = 5,
        category_id: Optional[str] = None,
        include_allergens: Optional[List[str]] = None,
        exclude_allergens: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self._ready:
            await self.ensure_collection()

        qvec = (await asyncio.to_thread(self._embed_sync, [query]))[0]
        qfilter = self._build_filter(
            category_id=category_id,
            include_allergens=include_allergens or [],
            exclude_allergens=exclude_allergens or [],
            min_price=min_price,
            max_price=max_price,
        )

        hits = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            query_vector=qvec,
            query_filter=qfilter,
            limit=max(1, min(int(top_k), 30)),
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for h in hits:
            p = h.payload or {}
            results.append(
                {
                    "menuItemId": p.get("menuItemId"),
                    "name": p.get("name"),
                    "categoryId": p.get("categoryId"),
                    "price": p.get("price"),
                    "allergies": p.get("allergies") or [],
                    "ingredients": p.get("ingredients") or [],
                    "score": float(getattr(h, "score", 0.0)),
                }
            )

        return {
            "collection": self.collection_name,
            "query": query,
            "count": len(results),
            "results": results,
        }

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        resp = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    def _build_embedding_text(self, item: Dict[str, Any]) -> str:
        name = str(item.get("name") or "")
        category = str(item.get("categoryId") or item.get("category") or "")
        price = item.get("price")
        ingredients = item.get("ingredients") or []
        allergies = item.get("allergies") or []
        description = str(item.get("description") or item.get("summary") or "")
        tags = self._build_tags(item)
        return (
            f"메뉴:{name}\n"
            f"카테고리:{category}\n"
            f"가격:{price}\n"
            f"재료:{', '.join(ingredients) if isinstance(ingredients, list) else ''}\n"
            f"알레르기:{', '.join(allergies) if isinstance(allergies, list) else ''}\n"
            f"태그:{', '.join(tags)}\n"
            f"설명:{description}"
        )

    def _build_payload(self, item: Dict[str, Any]) -> Dict[str, Any]:
        name = str(item.get("name") or "")
        category_id = str(item.get("categoryId") or item.get("category") or "")
        ingredients = item.get("ingredients") if isinstance(item.get("ingredients"), list) else []
        allergies = item.get("allergies") if isinstance(item.get("allergies"), list) else []
        price = item.get("price")
        try:
            price = float(price) if price is not None else None
        except Exception:
            price = None
        payload = {
            "menuItemId": str(item.get("menuItemId") or ""),
            "name": name,
            "categoryId": category_id,
            "price": price,
            "ingredients": [str(x) for x in ingredients if x is not None],
            "allergies": [str(x) for x in allergies if x is not None],
            "tags": self._build_tags(item),
            "description": str(item.get("description") or item.get("summary") or ""),
            "isSetLike": "세트" in name or category_id.lower().find("set") >= 0,
        }
        return payload

    def _build_tags(self, item: Dict[str, Any]) -> List[str]:
        tags = []
        name = str(item.get("name") or "")
        category = str(item.get("categoryId") or item.get("category") or "")
        if "세트" in name:
            tags.append("세트")
        if "버거" in name:
            tags.append("버거")
        if "음료" in name:
            tags.append("음료")
        if "사이드" in name:
            tags.append("사이드")
        if category:
            tags.append(category)
        return sorted(list(set(tags)))

    def _build_filter(
        self,
        category_id: Optional[str],
        include_allergens: List[str],
        exclude_allergens: List[str],
        min_price: Optional[float],
        max_price: Optional[float],
    ) -> Optional[qm.Filter]:
        must: List[Any] = []
        must_not: List[Any] = []

        if category_id:
            must.append(
                qm.FieldCondition(
                    key="categoryId",
                    match=qm.MatchValue(value=category_id),
                )
            )

        for a in include_allergens:
            must.append(
                qm.FieldCondition(
                    key="allergies",
                    match=qm.MatchValue(value=a),
                )
            )

        for a in exclude_allergens:
            must_not.append(
                qm.FieldCondition(
                    key="allergies",
                    match=qm.MatchValue(value=a),
                )
            )

        if min_price is not None or max_price is not None:
            must.append(
                qm.FieldCondition(
                    key="price",
                    range=qm.Range(
                        gte=min_price if min_price is not None else None,
                        lte=max_price if max_price is not None else None,
                    ),
                )
            )

        if not must and not must_not:
            return None
        return qm.Filter(must=must or None, must_not=must_not or None)

