"""Data Analysis Agent.

This agent validates whether a question is relevant to the selected database,
generates SQL through OpenAI or exact offline sample-question templates, checks
SQL safety, executes the query, and returns a structured analysis result.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.db.connector import SQLiteConnector
from src.db.schema_reader import SchemaReader
from src.db.sql_guard import ensure_limit


SQLGenerator = Callable[[str, str, str], str]

FALLBACK_SQL_TEMPLATES = {
    "chinook": {
        "show total sales by country": """
            SELECT BillingCountry AS country, ROUND(SUM(Total), 2) AS total_sales
            FROM invoices
            GROUP BY BillingCountry
            ORDER BY total_sales DESC
        """,
        "show invoice count by country": """
            SELECT BillingCountry AS country, COUNT(*) AS invoice_count
            FROM invoices
            GROUP BY BillingCountry
            ORDER BY invoice_count DESC
        """,
        "show artist sales ranking": """
            SELECT artists.Name AS artist, ROUND(SUM(invoice_items.UnitPrice * invoice_items.Quantity), 2) AS total_sales
            FROM artists
            JOIN albums ON artists.ArtistId = albums.ArtistId
            JOIN tracks ON albums.AlbumId = tracks.AlbumId
            JOIN invoice_items ON tracks.TrackId = invoice_items.TrackId
            GROUP BY artists.Name
            ORDER BY total_sales DESC
        """,
        "show top genres by track count": """
            SELECT genres.Name AS genre, COUNT(tracks.TrackId) AS track_count
            FROM genres
            JOIN tracks ON genres.GenreId = tracks.GenreId
            GROUP BY genres.Name
            ORDER BY track_count DESC
        """,
        "show customer count by country": """
            SELECT Country AS country, COUNT(*) AS customer_count
            FROM customers
            GROUP BY Country
            ORDER BY customer_count DESC
        """,
        "show monthly sales trend": """
            SELECT strftime('%Y-%m', InvoiceDate) AS month, ROUND(SUM(Total), 2) AS total_sales
            FROM invoices
            GROUP BY month
            ORDER BY month
        """,
    },
    "northwind": {
        "show top products by revenue": """
            SELECT Product.ProductName AS product,
                   ROUND(SUM(OrderDetail.UnitPrice * OrderDetail.Quantity * (1 - OrderDetail.Discount)), 2) AS revenue
            FROM OrderDetail
            JOIN Product ON Product.Id = OrderDetail.ProductId
            GROUP BY Product.ProductName
            ORDER BY revenue DESC
        """,
        "show order count by country": """
            SELECT ShipCountry AS country, COUNT(*) AS order_count
            FROM "Order"
            GROUP BY ShipCountry
            ORDER BY order_count DESC
        """,
        "show revenue by category": """
            SELECT Category.CategoryName AS category,
                   ROUND(SUM(OrderDetail.UnitPrice * OrderDetail.Quantity * (1 - OrderDetail.Discount)), 2) AS revenue
            FROM OrderDetail
            JOIN Product ON Product.Id = OrderDetail.ProductId
            JOIN Category ON Category.Id = Product.CategoryId
            GROUP BY Category.CategoryName
            ORDER BY revenue DESC
        """,
        "show revenue by customer": """
            SELECT Customer.CompanyName AS customer,
                   ROUND(SUM(OrderDetail.UnitPrice * OrderDetail.Quantity * (1 - OrderDetail.Discount)), 2) AS revenue
            FROM OrderDetail
            JOIN "Order" ON "Order".Id = OrderDetail.OrderId
            JOIN Customer ON Customer.Id = "Order".CustomerId
            GROUP BY Customer.CompanyName
            ORDER BY revenue DESC
        """,
        "show products with low stock": """
            SELECT ProductName AS product, UnitsInStock AS units_in_stock, ReorderLevel AS reorder_level
            FROM Product
            WHERE UnitsInStock <= ReorderLevel
            ORDER BY UnitsInStock ASC
        """,
        "show orders by employee": """
            SELECT Employee.FirstName || ' ' || Employee.LastName AS employee, COUNT("Order".Id) AS order_count
            FROM "Order"
            JOIN Employee ON Employee.Id = "Order".EmployeeId
            GROUP BY employee
            ORDER BY order_count DESC
        """,
    },
    "sakila": {
        "show revenue by category": """
            SELECT category.name AS category, ROUND(SUM(payment.amount), 2) AS revenue
            FROM payment
            JOIN rental ON rental.rental_id = payment.rental_id
            JOIN inventory ON inventory.inventory_id = rental.inventory_id
            JOIN film_category ON film_category.film_id = inventory.film_id
            JOIN category ON category.category_id = film_category.category_id
            GROUP BY category.name
            ORDER BY revenue DESC
        """,
        "show film rental count": """
            SELECT film.title AS film, COUNT(*) AS rental_count
            FROM rental
            JOIN inventory ON inventory.inventory_id = rental.inventory_id
            JOIN film ON film.film_id = inventory.film_id
            GROUP BY film.title
            ORDER BY rental_count DESC
        """,
        "show top customers by payment amount": """
            SELECT customer.first_name || ' ' || customer.last_name AS customer,
                   ROUND(SUM(payment.amount), 2) AS total_payment
            FROM payment
            JOIN customer ON customer.customer_id = payment.customer_id
            GROUP BY customer.customer_id
            ORDER BY total_payment DESC
        """,
        "show rentals by store": """
            SELECT store.store_id AS store, COUNT(rental.rental_id) AS rental_count
            FROM rental
            JOIN inventory ON inventory.inventory_id = rental.inventory_id
            JOIN store ON store.store_id = inventory.store_id
            GROUP BY store.store_id
            ORDER BY rental_count DESC
        """,
        "show actor film count": """
            SELECT actor.first_name || ' ' || actor.last_name AS actor, COUNT(film_actor.film_id) AS film_count
            FROM actor
            JOIN film_actor ON actor.actor_id = film_actor.actor_id
            GROUP BY actor.actor_id
            ORDER BY film_count DESC
        """,
        "show revenue by country": """
            SELECT country.country AS country, ROUND(SUM(payment.amount), 2) AS revenue
            FROM payment
            JOIN customer ON customer.customer_id = payment.customer_id
            JOIN address ON address.address_id = customer.address_id
            JOIN city ON city.city_id = address.city_id
            JOIN country ON country.country_id = city.country_id
            GROUP BY country.country
            ORDER BY revenue DESC
        """,
    },
}

FALLBACK_SAMPLE_QUESTIONS = {
    database_name: set(templates)
    for database_name, templates in FALLBACK_SQL_TEMPLATES.items()
}


# Normalizes sample questions so offline matching is stable.
def normalize_fallback_question(question: str) -> str:
    return " ".join(question.lower().split())


# Checks whether offline mode supports a database/question pair.
def is_supported_fallback_question(database_name: str, question: str) -> bool:
    normalized_question = normalize_fallback_question(question)
    return normalized_question in FALLBACK_SQL_TEMPLATES.get(database_name, {})


class AnalysisAgentError(RuntimeError):
    """Base class for analysis agent failures."""


class SQLGenerationError(AnalysisAgentError):
    """Raised when the agent cannot produce a usable SQL query."""


class InvalidAnalysisQuestionError(AnalysisAgentError):
    """Raised when the user's request is not a database analysis question."""


@dataclass
class AnalysisResult:
    database_name: str
    question: str
    sql: str
    summary: str
    dataframe: pd.DataFrame
    sql_source: str

    # Returns the number of rows produced by the analysis query.
    @property
    def row_count(self) -> int:
        return len(self.dataframe)


class DataAnalysisAgent:
    # Configures dependencies, model choice, row limit, and online/offline mode.
    def __init__(
        self,
        connector: SQLiteConnector | None = None,
        schema_reader: SchemaReader | None = None,
        model: str | None = None,
        row_limit: int = 100,
        use_openai: bool = True,
        sql_generator: SQLGenerator | None = None,
    ) -> None:
        load_dotenv()
        self.connector = connector or SQLiteConnector()
        self.schema_reader = schema_reader or SchemaReader(self.connector)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.row_limit = row_limit
        self.use_openai = use_openai
        self.sql_generator = sql_generator

    # Runs the full analysis flow from question to DataFrame and summary.
    def analyze(
        self,
        database_path: str | Path,
        database_name: str,
        question: str,
    ) -> AnalysisResult:
        schema_text = self.schema_reader.get_schema_text(database_path)
        self._validate_question_relevance(question, schema_text)
        generated_sql, sql_source = self.generate_sql(database_name, question, schema_text)
        safe_sql = ensure_limit(generated_sql, limit=self.row_limit)
        dataframe = self.connector.run_query(database_path, safe_sql)
        summary = self.summarize(question, dataframe)

        return AnalysisResult(
            database_name=database_name,
            question=question,
            sql=safe_sql,
            summary=summary,
            dataframe=dataframe,
            sql_source=sql_source,
        )

    # Chooses injected SQL, OpenAI SQL, or offline sample-template SQL.
    def generate_sql(
        self,
        database_name: str,
        question: str,
        schema_text: str,
    ) -> tuple[str, str]:
        if self.sql_generator is not None:
            return self.sql_generator(database_name, question, schema_text), "injected"

        if self.use_openai and os.getenv("OPENAI_API_KEY"):
            try:
                return self._generate_sql_with_openai(database_name, question, schema_text), "openai"
            except Exception:
                fallback_sql = self._generate_fallback_sql(database_name, question, schema_text)
                return fallback_sql, "fallback"

        return self._generate_fallback_sql(database_name, question, schema_text), "fallback"

    # Calls OpenAI to generate a SQLite SELECT query from schema and question.
    def _generate_sql_with_openai(
        self,
        database_name: str,
        question: str,
        schema_text: str,
    ) -> str:
        from openai import OpenAI

        client = OpenAI()
        prompt = self._build_sql_prompt(database_name, question, schema_text)
        response = client.responses.create(
            model=self.model,
            input=prompt,
        )
        return self._extract_sql(response.output_text)

    # Builds the prompt that constrains the model to safe SQL output.
    def _build_sql_prompt(self, database_name: str, question: str, schema_text: str) -> str:
        return f"""
You are the Data Analysis Agent for a SQLite analytics system.

Database name:
{database_name}

SQLite schema:
{schema_text}

User question:
{question}

Return exactly one SQLite SELECT query.
Rules:
- Return SQL only.
- Do not use markdown.
- Do not explain the query.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, or VACUUM.
- Use double quotes around table names or column names only when needed.
- Prefer clear aliases for output columns.
""".strip()

    # Extracts the first SELECT/WITH query from a model response.
    def _extract_sql(self, text: str) -> str:
        code_block = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        candidate = code_block.group(1).strip() if code_block else text.strip()
        candidate = candidate.strip("`").strip()

        match = re.search(r"\b(WITH|SELECT)\b", candidate, flags=re.IGNORECASE)
        if not match:
            raise SQLGenerationError(f"No SELECT query found in model response: {text}")

        return candidate[match.start() :].strip()

    # Looks up exact offline SQL templates for supported sample questions.
    def _generate_fallback_sql(
        self,
        database_name: str,
        question: str,
        schema_text: str,
    ) -> str:
        normalized_question = normalize_fallback_question(question)
        templates = FALLBACK_SQL_TEMPLATES.get(database_name, {})

        try:
            return templates[normalized_question]
        except KeyError as exc:
            raise SQLGenerationError(
                "Offline fallback only supports the predefined analysis sample questions. "
                "Please choose one from Example question, or switch to Online OpenAI for custom questions."
            ) from exc

    # Wraps the module-level fallback support check for class callers.
    def _is_supported_fallback_question(self, database_name: str, normalized_question: str) -> bool:
        return is_supported_fallback_question(database_name, normalized_question)

    # Rejects questions that do not look related to the selected database.
    def _validate_question_relevance(self, question: str, schema_text: str) -> None:
        # The guard prevents non-database requests from being forced into SQL.
        normalized_question = question.lower()
        question_words = set(re.findall(r"[a-z][a-z0-9_]+", normalized_question))
        schema_words = self._schema_keywords(schema_text)
        analysis_terms = {
            "actor",
            "amount",
            "average",
            "avg",
            "bottom",
            "category",
            "compare",
            "count",
            "country",
            "customer",
            "database",
            "employee",
            "film",
            "genre",
            "group",
            "invoice",
            "list",
            "low",
            "monthly",
            "order",
            "payment",
            "product",
            "rank",
            "ranking",
            "records",
            "rental",
            "revenue",
            "rows",
            "sales",
            "show",
            "stock",
            "store",
            "sum",
            "table",
            "top",
            "total",
            "track",
            "trend",
        }

        has_analysis_intent = bool(question_words.intersection(analysis_terms))
        analysis_term_count = len(question_words.intersection(analysis_terms))
        has_schema_overlap = bool(question_words.intersection(schema_words))
        has_generic_data_language = bool(
            question_words.intersection({"database", "data", "records", "rows", "table"})
        )

        if has_analysis_intent and (
            has_schema_overlap or has_generic_data_language or analysis_term_count >= 2
        ):
            return

        raise InvalidAnalysisQuestionError(
            "This does not look like a database analysis question for the selected database. "
            "Please ask about tables, fields, counts, totals, rankings, revenue, sales, orders, customers, rentals, or trends."
        )

    # Extracts useful table and column words from schema text.
    def _schema_keywords(self, schema_text: str) -> set[str]:
        spaced_schema = re.sub(r"([a-z])([A-Z])", r"\1 \2", schema_text)
        words = {
            word.lower()
            for word in re.findall(r"[A-Za-z][A-Za-z0-9_]+", spaced_schema)
            if len(word) >= 3
        }
        ignored = {
            "blob",
            "char",
            "date",
            "datetime",
            "decimal",
            "double",
            "integer",
            "nvarchar",
            "numeric",
            "smallint",
            "table",
            "text",
            "timestamp",
            "varchar",
        }
        keywords = words - ignored
        singulars = {word[:-1] for word in keywords if word.endswith("s") and len(word) > 3}
        return keywords.union(singulars)

    # Finds the first schema table name for legacy fallback helpers.
    def _first_table_from_schema(self, schema_text: str) -> str:
        match = re.search(r"^Table\s+([^:]+):", schema_text, flags=re.MULTILINE)
        if not match:
            raise SQLGenerationError("Cannot infer a fallback table from schema.")

        return match.group(1)

    # Creates a compact human-readable summary of the returned DataFrame.
    def summarize(self, question: str, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return "The query ran successfully but returned no rows."

        columns = ", ".join(str(column) for column in dataframe.columns)
        first_row = dataframe.iloc[0].to_dict()
        first_row_text = ", ".join(f"{key}={value}" for key, value in first_row.items())

        return (
            f"Returned {len(dataframe)} rows for: {question}. "
            f"Columns: {columns}. Top result: {first_row_text}."
        )
