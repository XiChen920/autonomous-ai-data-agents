"""Semantic metadata index for database, table, column, and sample-question retrieval.

The index turns schema metadata into embedding vectors, embeds the user's
question, and returns only the most relevant schema context for SQL generation.
It uses a deterministic local embedding provider by default so tests and offline
demo runs do not require an API call. If needed, the provider can be switched to
OpenAI embeddings through SCHEMA_EMBEDDING_PROVIDER=openai.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from src.db.registry import DatabaseRegistry
from src.db.schema_reader import SchemaReader


DOCUMENT_DATABASE = "database"
DOCUMENT_TABLE = "table"
DOCUMENT_COLUMN = "column"
DOCUMENT_SAMPLE_QUESTION = "sample_question"


TOKEN_SYNONYMS = {
    "amount": ("amount", "revenue", "sales", "total", "payment"),
    "billing": ("billing", "invoice", "customer", "country"),
    "buy": ("buy", "order", "sale", "purchase"),
    "category": ("category", "genre", "group"),
    "categories": ("category", "genre", "group"),
    "country": ("country", "nation", "location", "billing"),
    "customer": ("customer", "client", "buyer"),
    "customers": ("customer", "client", "buyer"),
    "dvd": ("dvd", "film", "movie", "rental"),
    "film": ("film", "movie", "rental", "dvd"),
    "films": ("film", "movie", "rental", "dvd"),
    "genre": ("genre", "category", "music"),
    "genres": ("genre", "category", "music"),
    "invoice": ("invoice", "sales", "revenue", "billing"),
    "invoices": ("invoice", "sales", "revenue", "billing"),
    "month": ("month", "monthly", "date", "trend"),
    "monthly": ("month", "monthly", "date", "trend"),
    "movie": ("movie", "film", "rental", "dvd"),
    "order": ("order", "purchase", "sales", "revenue"),
    "orders": ("order", "purchase", "sales", "revenue"),
    "payment": ("payment", "amount", "revenue", "sales"),
    "payments": ("payment", "amount", "revenue", "sales"),
    "product": ("product", "item", "stock", "inventory"),
    "products": ("product", "item", "stock", "inventory"),
    "rental": ("rental", "film", "movie", "dvd"),
    "rentals": ("rental", "film", "movie", "dvd"),
    "sale": ("sale", "sales", "revenue", "amount", "total"),
    "sales": ("sale", "sales", "revenue", "amount", "total"),
    "stock": ("stock", "inventory", "product"),
    "store": ("store", "shop", "location"),
    "total": ("total", "sum", "amount", "revenue", "sales"),
    "trend": ("trend", "month", "date", "time"),
}


@dataclass(frozen=True)
class MetadataDocument:
    """One searchable metadata document and its embedding vector."""

    document_id: str
    database_name: str
    document_type: str
    text: str
    table_name: str | None = None
    column_name: str | None = None
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class RetrievedMetadata:
    """One retrieved metadata document plus its similarity score."""

    document: MetadataDocument
    score: float


@dataclass(frozen=True)
class DatabaseCandidate:
    """Database-level retrieval result for future database auto-routing."""

    database_name: str
    score: float
    evidence: tuple[RetrievedMetadata, ...]


@dataclass(frozen=True)
class SchemaRetrievalResult:
    """Compact schema context selected for one user question."""

    database_name: str
    database_description: str
    schema_text: str
    retrieved_tables: tuple[str, ...]
    retrieved_columns: tuple[str, ...]
    retrieved_sample_questions: tuple[str, ...]
    hits: tuple[RetrievedMetadata, ...]


class EmbeddingProvider(Protocol):
    """Protocol for local or hosted text embedding providers."""

    # Converts text inputs into embedding vectors with matching order.
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class HashingEmbeddingProvider:
    """Deterministic local embedding provider used for tests and offline runs."""

    # Configures vector dimension for the hashed bag-of-words representation.
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    # Converts texts into normalized hashed token vectors.
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    # Embeds one text using expanded schema-aware tokens.
    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in expand_tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    """Hosted embedding provider for larger or more semantic deployments."""

    # Configures the OpenAI embedding model and optional injected client.
    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = client or OpenAI()

    # Calls the OpenAI embeddings endpoint and returns vectors in input order.
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]


# Creates the configured embedding provider.
def create_embedding_provider(provider_name: str | None = None) -> EmbeddingProvider:
    name = (provider_name or os.getenv("SCHEMA_EMBEDDING_PROVIDER", "local")).lower()
    if name == "openai":
        return OpenAIEmbeddingProvider()

    return HashingEmbeddingProvider()


# Splits identifiers and prose into normalized retrieval tokens.
def tokenize(text: str) -> list[str]:
    spaced_text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", spaced_text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.append(token)
        if token.endswith("s") and len(token) > 3:
            tokens.append(token[:-1])

    return tokens


# Adds synonym/concept tokens so local embeddings behave more semantically.
def expand_tokens(text: str) -> list[str]:
    expanded: list[str] = []
    for token in tokenize(text):
        expanded.append(token)
        expanded.extend(TOKEN_SYNONYMS.get(token, ()))

    bigrams = [
        f"{left}_{right}"
        for left, right in zip(expanded, expanded[1:])
        if left != right
    ]
    return expanded + bigrams


# Calculates cosine similarity between two normalized embedding vectors.
def cosine_similarity(left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]) -> float:
    if not left or not right:
        return 0.0

    return sum(left_value * right_value for left_value, right_value in zip(left, right))


class SchemaMetadataIndex:
    """In-memory vector index for schema metadata documents."""

    # Stores embedded documents and the provider used for query embeddings.
    def __init__(
        self,
        documents: list[MetadataDocument],
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or create_embedding_provider()
        self.documents = documents

    # Builds and embeds documents in one batch.
    @classmethod
    def build(
        cls,
        documents: list[MetadataDocument],
        embedding_provider: EmbeddingProvider | None = None,
    ) -> "SchemaMetadataIndex":
        provider = embedding_provider or create_embedding_provider()
        embeddings = provider.embed_texts([document.text for document in documents])
        embedded_documents = [
            replace(document, embedding=tuple(embedding))
            for document, embedding in zip(documents, embeddings)
        ]
        return cls(embedded_documents, provider)

    # Searches documents by embedding similarity with optional filters.
    def search(
        self,
        query: str,
        top_k: int = 10,
        document_types: set[str] | None = None,
        database_names: set[str] | None = None,
    ) -> list[RetrievedMetadata]:
        query_embedding = self.embedding_provider.embed_texts([query])[0]
        return self.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            document_types=document_types,
            database_names=database_names,
        )

    # Searches documents with an already-computed query embedding.
    def search_by_embedding(
        self,
        query_embedding: list[float] | tuple[float, ...],
        top_k: int = 10,
        document_types: set[str] | None = None,
        database_names: set[str] | None = None,
    ) -> list[RetrievedMetadata]:
        hits: list[RetrievedMetadata] = []

        for document in self.documents:
            if document_types is not None and document.document_type not in document_types:
                continue
            if database_names is not None and document.database_name not in database_names:
                continue

            score = cosine_similarity(query_embedding, document.embedding)
            hits.append(RetrievedMetadata(document=document, score=score))

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


class SemanticSchemaRetriever:
    """Builds metadata indexes and retrieves compact schema context for questions."""

    # Configures registry, schema reader, embeddings, retrieval sizes, and cache.
    def __init__(
        self,
        database_registry: DatabaseRegistry | None = None,
        schema_reader: SchemaReader | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        top_tables: int = 6,
        top_columns: int = 40,
        top_sample_questions: int = 5,
    ) -> None:
        self.database_registry = database_registry
        self.schema_reader = schema_reader or SchemaReader()
        self.embedding_provider = embedding_provider or create_embedding_provider()
        self.top_tables = top_tables
        self.top_columns = top_columns
        self.top_sample_questions = top_sample_questions
        self._database_index_cache: dict[tuple[Any, ...], SchemaMetadataIndex] = {}

    # Retrieves relevant context inside one selected database.
    def retrieve(
        self,
        database_name: str,
        database_path: str | Path,
        question: str,
        database_description: str = "",
        sample_questions: list[str] | tuple[str, ...] | None = None,
    ) -> SchemaRetrievalResult:
        index = self._index_for_database(
            database_name=database_name,
            database_path=database_path,
            database_description=database_description,
            sample_questions=sample_questions or (),
        )

        question_embedding = index.embedding_provider.embed_texts([question])[0]
        table_hits = index.search_by_embedding(
            question_embedding,
            top_k=self.top_tables,
            document_types={DOCUMENT_TABLE},
        )
        column_hits = index.search_by_embedding(
            question_embedding,
            top_k=self.top_columns,
            document_types={DOCUMENT_COLUMN},
        )
        sample_hits = index.search_by_embedding(
            question_embedding,
            top_k=self.top_sample_questions,
            document_types={DOCUMENT_SAMPLE_QUESTION},
        )
        database_hits = index.search_by_embedding(
            question_embedding,
            top_k=1,
            document_types={DOCUMENT_DATABASE},
        )

        table_names = self._rank_table_names(table_hits, column_hits)
        if not table_names:
            table_names = tuple(self.schema_reader.list_tables(database_path)[: self.top_tables])

        column_names = tuple(
            self._document_column_label(hit.document)
            for hit in column_hits
            if hit.document.table_name and hit.document.column_name
        )
        sample_question_texts = tuple(hit.document.text for hit in sample_hits)
        schema_text = self._build_retrieved_schema_text(
            database_name=database_name,
            database_path=database_path,
            database_description=database_description,
            table_names=table_names,
            column_names=column_names,
            sample_questions=sample_question_texts,
        )

        return SchemaRetrievalResult(
            database_name=database_name,
            database_description=database_description,
            schema_text=schema_text,
            retrieved_tables=table_names,
            retrieved_columns=column_names,
            retrieved_sample_questions=sample_question_texts,
            hits=tuple(database_hits + table_hits + column_hits + sample_hits),
        )

    # Scores all configured databases so a future router can choose a database automatically.
    def retrieve_database_candidates(
        self,
        question: str,
        database_names: list[str] | tuple[str, ...] | None = None,
        sample_questions_by_database: dict[str, list[str] | tuple[str, ...]] | None = None,
        top_k: int = 5,
    ) -> list[DatabaseCandidate]:
        if self.database_registry is None:
            self.database_registry = DatabaseRegistry()

        names = list(database_names or self.database_registry.list_databases())
        documents: list[MetadataDocument] = []
        for database_name in names:
            database_info = self.database_registry.get_database(database_name)
            documents.extend(
                build_database_metadata_documents(
                    database_name=database_name,
                    database_path=database_info["path"],
                    database_description=str(database_info.get("description", "")),
                    schema_reader=self.schema_reader,
                    sample_questions=(sample_questions_by_database or {}).get(database_name, ()),
                )
            )

        index = SchemaMetadataIndex.build(documents, self.embedding_provider)
        question_embedding = index.embedding_provider.embed_texts([question])[0]
        hits = index.search_by_embedding(
            question_embedding,
            top_k=max(top_k * 6, 20),
        )
        evidence_by_database: dict[str, list[RetrievedMetadata]] = defaultdict(list)
        score_by_database: dict[str, float] = defaultdict(float)

        for hit in hits:
            database_name = hit.document.database_name
            evidence_by_database[database_name].append(hit)
            score_by_database[database_name] += hit.score

        candidates = [
            DatabaseCandidate(
                database_name=database_name,
                score=score,
                evidence=tuple(evidence_by_database[database_name][:5]),
            )
            for database_name, score in score_by_database.items()
        ]
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:top_k]

    # Returns a cached metadata index for one database file and sample-question set.
    def _index_for_database(
        self,
        database_name: str,
        database_path: str | Path,
        database_description: str,
        sample_questions: list[str] | tuple[str, ...],
    ) -> SchemaMetadataIndex:
        path = Path(database_path).resolve()
        cache_key = (
            database_name,
            str(path),
            path.stat().st_mtime if path.exists() else 0,
            database_description,
            tuple(sorted(sample_questions)),
        )
        if cache_key not in self._database_index_cache:
            documents = build_database_metadata_documents(
                database_name=database_name,
                database_path=path,
                database_description=database_description,
                schema_reader=self.schema_reader,
                sample_questions=sample_questions,
            )
            self._database_index_cache[cache_key] = SchemaMetadataIndex.build(
                documents,
                self.embedding_provider,
            )

        return self._database_index_cache[cache_key]

    # Combines table and column hits into a stable ordered table list.
    def _rank_table_names(
        self,
        table_hits: list[RetrievedMetadata],
        column_hits: list[RetrievedMetadata],
    ) -> tuple[str, ...]:
        scores: dict[str, float] = defaultdict(float)
        for hit in table_hits:
            if hit.document.table_name:
                scores[hit.document.table_name] += hit.score

        for hit in column_hits:
            if hit.document.table_name:
                scores[hit.document.table_name] += hit.score * 0.65

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return tuple(table_name for table_name, _score in ranked[: self.top_tables])

    # Formats a table.column label for retrieved columns.
    def _document_column_label(self, document: MetadataDocument) -> str:
        return f"{document.table_name}.{document.column_name}"

    # Builds the compact schema text that is sent to the SQL-generation step.
    def _build_retrieved_schema_text(
        self,
        database_name: str,
        database_path: str | Path,
        database_description: str,
        table_names: tuple[str, ...],
        column_names: tuple[str, ...],
        sample_questions: tuple[str, ...],
    ) -> str:
        lines = [
            "Semantic schema retrieval result.",
            f"Database {database_name}: {database_description or 'No description provided.'}",
        ]

        if sample_questions:
            lines.append("Relevant sample-question metadata:")
            lines.extend(f"- {question}" for question in sample_questions)

        lines.append("Relevant tables and columns:")
        relevant_column_set = set(column_names)
        for table_name in table_names:
            columns = self.schema_reader.get_table_columns(database_path, table_name)
            column_text = ", ".join(
                f"{column['name']} {column['type']}".strip()
                for column in columns
            )
            highlighted_columns = [
                label.split(".", 1)[1]
                for label in column_names
                if label.startswith(f"{table_name}.")
            ]
            lines.append(f"Table {table_name}: {column_text}")
            if highlighted_columns:
                lines.append(
                    f"Most relevant columns in {table_name}: {', '.join(highlighted_columns)}"
                )

        if relevant_column_set:
            lines.append(f"Retrieved column labels: {', '.join(column_names)}")

        return "\n".join(lines)


# Builds metadata documents for one SQLite database.
def build_database_metadata_documents(
    database_name: str,
    database_path: str | Path,
    database_description: str,
    schema_reader: SchemaReader,
    sample_questions: list[str] | tuple[str, ...] = (),
) -> list[MetadataDocument]:
    tables = schema_reader.list_tables(database_path)
    documents: list[MetadataDocument] = [
        MetadataDocument(
            document_id=f"{database_name}:database",
            database_name=database_name,
            document_type=DOCUMENT_DATABASE,
            text=(
                f"Database {database_name}. Description: {database_description}. "
                f"Tables: {', '.join(tables)}. "
                f"Sample questions: {', '.join(sample_questions)}."
            ),
        )
    ]

    for table_name in tables:
        columns = schema_reader.get_table_columns(database_path, table_name)
        column_descriptions = [
            describe_column(table_name, column)
            for column in columns
        ]
        documents.append(
            MetadataDocument(
                document_id=f"{database_name}:table:{table_name}",
                database_name=database_name,
                document_type=DOCUMENT_TABLE,
                table_name=table_name,
                text=(
                    f"Database {database_name}. Table {table_name}. "
                    f"Columns: {', '.join(column['name'] for column in columns)}. "
                    f"Column descriptions: {' '.join(column_descriptions)}"
                ),
            )
        )

        for column in columns:
            column_name = str(column["name"])
            documents.append(
                MetadataDocument(
                    document_id=f"{database_name}:column:{table_name}.{column_name}",
                    database_name=database_name,
                    document_type=DOCUMENT_COLUMN,
                    table_name=table_name,
                    column_name=column_name,
                    text=(
                        f"Database {database_name}. Table {table_name}. "
                        f"Column {column_name}. Type {column.get('type', '')}. "
                        f"Description: {describe_column(table_name, column)}"
                    ),
                )
            )

    for index, question in enumerate(sample_questions, start=1):
        documents.append(
            MetadataDocument(
                document_id=f"{database_name}:sample_question:{index}",
                database_name=database_name,
                document_type=DOCUMENT_SAMPLE_QUESTION,
                text=f"Sample question for database {database_name}: {question}",
            )
        )

    return documents


# Creates a plain-English description from SQLite column metadata.
def describe_column(table_name: str, column: dict[str, Any]) -> str:
    column_name = str(column["name"])
    column_words = " ".join(tokenize(column_name))
    constraints = []
    if column.get("primary_key"):
        constraints.append("primary key")
    if column.get("not_null"):
        constraints.append("not null")
    constraint_text = f" It is {', '.join(constraints)}." if constraints else ""

    return (
        f"{column_name} is a column on table {table_name}. "
        f"Name words: {column_words}. SQLite type: {column.get('type', '')}.{constraint_text}"
    )
