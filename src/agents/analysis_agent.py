"""Data Analysis Agent.

This agent validates whether a question is relevant to the selected database,
retrieves the most relevant schema metadata, generates SQL through OpenAI or
exact offline sample-question templates, checks SQL safety, executes the query,
and returns a structured analysis result.
"""

import os
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.db.connector import SQLiteConnector
from src.db.schema_reader import SchemaReader
from src.db.sql_guard import ensure_limit
from src.retrieval.schema_metadata_index import SemanticSchemaRetriever


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
class SQLGenerationMetrics:
    tool_call_count: int = 0
    sql_execution_attempts: int = 0
    sql_execution_failures: int = 0
    sql_execution_successes: int = 0

    # Counts SQL execution failures that required the model to repair a query.
    @property
    def sql_repair_count(self) -> int:
        return self.sql_execution_failures

    # Reports whether at least one failed SQL attempt was later repaired successfully.
    @property
    def sql_repair_succeeded(self) -> bool:
        return self.sql_execution_failures > 0 and self.sql_execution_successes > 0


@dataclass
class AnalysisResult:
    database_name: str
    question: str
    sql: str
    summary: str
    dataframe: pd.DataFrame
    sql_source: str
    schema_context: str = ""
    retrieved_tables: tuple[str, ...] = ()
    retrieved_columns: tuple[str, ...] = ()
    retrieved_sample_questions: tuple[str, ...] = ()
    tool_call_count: int = 0
    sql_execution_attempts: int = 0
    sql_execution_failures: int = 0
    sql_repair_count: int = 0
    sql_repair_succeeded: bool = False

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
        openai_client: Any | None = None,
        schema_retriever: SemanticSchemaRetriever | None = None,
        max_tool_iterations: int = 6,
    ) -> None:
        load_dotenv()
        self.connector = connector or SQLiteConnector()
        self.schema_reader = schema_reader or SchemaReader(self.connector)
        self.schema_retriever = schema_retriever or SemanticSchemaRetriever(
            schema_reader=self.schema_reader
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.row_limit = row_limit
        self.use_openai = use_openai
        self.sql_generator = sql_generator
        self.openai_client = openai_client
        self.max_tool_iterations = max_tool_iterations
        self._sql_generation_metrics = SQLGenerationMetrics()

    # Runs the full analysis flow from question to DataFrame and summary.
    def analyze(
        self,
        database_path: str | Path,
        database_name: str,
        question: str,
        database_description: str = "",
    ) -> AnalysisResult:
        full_schema_text = self.schema_reader.get_schema_text(database_path)
        retrieval_result = self.schema_retriever.retrieve(
            database_name=database_name,
            database_path=database_path,
            question=question,
            database_description=database_description,
            sample_questions=sorted(FALLBACK_SQL_TEMPLATES.get(database_name, {})),
        )
        schema_text = retrieval_result.schema_text
        self._validate_question_relevance(
            question,
            f"{full_schema_text}\n{schema_text}",
        )
        generated_sql, sql_source = self.generate_sql(
            database_name,
            question,
            schema_text,
            database_path=database_path,
        )
        metrics = self._sql_generation_metrics
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
            schema_context=schema_text,
            retrieved_tables=retrieval_result.retrieved_tables,
            retrieved_columns=retrieval_result.retrieved_columns,
            retrieved_sample_questions=retrieval_result.retrieved_sample_questions,
            tool_call_count=metrics.tool_call_count,
            sql_execution_attempts=metrics.sql_execution_attempts,
            sql_execution_failures=metrics.sql_execution_failures,
            sql_repair_count=metrics.sql_repair_count,
            sql_repair_succeeded=metrics.sql_repair_succeeded,
        )

    # Chooses injected SQL, OpenAI SQL, or offline sample-template SQL.
    def generate_sql(
        self,
        database_name: str,
        question: str,
        schema_text: str,
        database_path: str | Path | None = None,
    ) -> tuple[str, str]:
        self._sql_generation_metrics = SQLGenerationMetrics()
        if self.sql_generator is not None:
            return self.sql_generator(database_name, question, schema_text), "injected"

        if self.use_openai and os.getenv("OPENAI_API_KEY"):
            try:
                return (
                    self._generate_sql_with_openai(
                        database_path,
                        database_name,
                        question,
                        schema_text,
                    ),
                    "openai",
                )
            except Exception as openai_error:
                try:
                    self._sql_generation_metrics = SQLGenerationMetrics()
                    fallback_sql = self._generate_fallback_sql(database_name, question, schema_text)
                    return fallback_sql, "fallback"
                except SQLGenerationError as fallback_error:
                    raise SQLGenerationError(
                        "OpenAI SQL generation failed and no offline fallback template matched. "
                        f"OpenAI error: {openai_error}"
                    ) from fallback_error

        return self._generate_fallback_sql(database_name, question, schema_text), "fallback"

    # Runs an OpenAI tool-calling loop over retrieved schema context.
    def _generate_sql_with_openai(
        self,
        database_path: str | Path | None,
        database_name: str,
        question: str,
        schema_text: str,
    ) -> str:
        from openai import OpenAI

        if database_path is None:
            raise SQLGenerationError("database_path is required for OpenAI tool-calling SQL generation.")

        client = self.openai_client or OpenAI()
        tools = self._build_analysis_tools()
        instructions = self._build_tool_calling_instructions()
        prompt = self._build_tool_calling_prompt(database_name, question)
        conversation_input: list[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]
        response = client.responses.create(
            model=self.model,
            instructions=instructions,
            input=conversation_input,
            tools=tools,
            parallel_tool_calls=False,
        )

        last_error = ""
        for _iteration in range(self.max_tool_iterations):
            tool_calls = self._response_function_calls(response)
            if tool_calls:
                tool_outputs = [
                    self._execute_tool_call(
                        tool_call,
                        database_path=database_path,
                        database_name=database_name,
                        schema_text=schema_text,
                    )
                    for tool_call in tool_calls
                ]
                conversation_input.extend(self._response_output_as_input_items(response))
                conversation_input.extend(tool_outputs)
                response = client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=conversation_input,
                    tools=tools,
                    parallel_tool_calls=False,
                )
                continue

            final_sql = self._extract_sql(response.output_text)
            preflight_result = self._run_sql_tool(database_path, final_sql)
            if preflight_result["ok"]:
                return final_sql

            last_error = str(preflight_result["error"])
            conversation_input.append(
                {
                    "role": "user",
                    "content": self._build_repair_prompt(final_sql, preflight_result),
                }
            )
            response = client.responses.create(
                model=self.model,
                instructions=instructions,
                input=conversation_input,
                tools=tools,
                parallel_tool_calls=False,
            )

        raise SQLGenerationError(
            "OpenAI tool-calling loop did not produce executable SQL. "
            f"Last error: {last_error or 'no final SQL returned'}"
        )

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

    # Defines the local analysis tools available to the OpenAI model.
    def _build_analysis_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "get_schema",
                "description": "Return the semantically retrieved SQLite schema context as compact text.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "validate_sql",
                "description": "Validate that a SQL query is a safe read-only SQLite SELECT/WITH query and add LIMIT if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "The SQLite SQL query to validate.",
                        }
                    },
                    "required": ["sql"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "run_sql",
                "description": "Execute a safe read-only SQLite query and return row count, columns, and preview rows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "The SQLite SELECT/WITH query to execute.",
                        }
                    },
                    "required": ["sql"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

    # Builds developer instructions for the tool-calling SQL agent loop.
    def _build_tool_calling_instructions(self) -> str:
        return """
You are the Data Analysis Agent for a SQLite analytics system.
Use the provided tools to inspect the retrieved schema context, validate SQL, and test execution.
Before returning a final answer, call run_sql successfully at least once.
If run_sql returns an error, revise the SQL and call run_sql again.
Return exactly one final SQLite SELECT/WITH query after a successful run_sql call.
Rules:
- Return SQL only in the final answer.
- Do not use markdown.
- Do not explain the query.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, or VACUUM.
- Prefer clear aliases for output columns.
""".strip()

    # Builds the user prompt for the OpenAI tool-calling loop.
    def _build_tool_calling_prompt(self, database_name: str, question: str) -> str:
        return f"""
Database name:
{database_name}

User question:
{question}

Use get_schema first, then generate and test a safe SQLite query.
The schema tool returns only the database/table/column metadata retrieved as relevant to this question.
""".strip()

    # Sends final-SQL execution failures back to the model for repair.
    def _build_repair_prompt(self, final_sql: str, preflight_result: dict[str, Any]) -> str:
        return f"""
The final SQL failed preflight execution.

SQL:
{final_sql}

Error:
{preflight_result['error']}

Revise the SQL using the available tools. Call run_sql again before returning final SQL.
""".strip()

    # Extracts function calls from a Responses API result.
    def _response_function_calls(self, response) -> list[Any]:
        output_items = getattr(response, "output", None) or []
        return [
            item
            for item in output_items
            if self._tool_item_value(item, "type") == "function_call"
        ]

    # Converts response output items into stateless input items for ZDR-compatible loops.
    def _response_output_as_input_items(self, response) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        for item in getattr(response, "output", None) or []:
            item_type = self._tool_item_value(item, "type")
            if item_type != "function_call":
                continue

            input_item = {
                "type": "function_call",
                "call_id": self._tool_item_value(item, "call_id", ""),
                "name": self._tool_item_value(item, "name", ""),
                "arguments": self._tool_item_value(item, "arguments", "{}"),
            }
            item_id = self._tool_item_value(item, "id")
            status = self._tool_item_value(item, "status")
            if item_id:
                input_item["id"] = item_id
            if status:
                input_item["status"] = status

            input_items.append(input_item)

        return input_items

    # Handles both SDK objects and lightweight dicts used in tests.
    def _tool_item_value(self, item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    # Executes one model-requested analysis tool and formats output for the next model turn.
    def _execute_tool_call(
        self,
        tool_call: Any,
        database_path: str | Path,
        database_name: str,
        schema_text: str,
    ) -> dict[str, str]:
        tool_name = self._tool_item_value(tool_call, "name", "")
        call_id = self._tool_item_value(tool_call, "call_id", "")
        arguments_text = self._tool_item_value(tool_call, "arguments", "{}")
        self._sql_generation_metrics.tool_call_count += 1

        try:
            arguments = self._parse_tool_arguments(arguments_text)
            output = self._execute_analysis_tool(
                tool_name,
                arguments,
                database_path=database_path,
                database_name=database_name,
                schema_text=schema_text,
            )
        except Exception as exc:
            output = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(output, ensure_ascii=False),
        }

    # Parses JSON arguments from a model tool call.
    def _parse_tool_arguments(self, arguments_text: str) -> dict[str, Any]:
        arguments = json.loads(arguments_text or "{}")
        if not isinstance(arguments, dict):
            raise SQLGenerationError("Tool arguments must be a JSON object.")
        return arguments

    # Dispatches local tool implementations for schema, validation, and execution.
    def _execute_analysis_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        database_path: str | Path,
        database_name: str,
        schema_text: str,
    ) -> dict[str, Any]:
        if tool_name == "get_schema":
            return {
                "ok": True,
                "database_name": database_name,
                "schema_text": schema_text,
            }

        if tool_name == "validate_sql":
            sql = self._required_sql_argument(arguments)
            return self._validate_sql_tool(sql)

        if tool_name == "run_sql":
            sql = self._required_sql_argument(arguments)
            return self._run_sql_tool(database_path, sql)

        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    # Reads and validates the required SQL argument for SQL tools.
    def _required_sql_argument(self, arguments: dict[str, Any]) -> str:
        sql = arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise SQLGenerationError("Tool argument 'sql' is required.")
        return sql

    # Runs only SQL safety validation and reports errors as tool output.
    def _validate_sql_tool(self, sql: str) -> dict[str, Any]:
        try:
            safe_sql = ensure_limit(sql, limit=self.row_limit)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "sql": sql}

        return {"ok": True, "safe_sql": safe_sql}

    # Executes SQL as a tool call and returns a small observation for the model.
    def _run_sql_tool(self, database_path: str | Path, sql: str) -> dict[str, Any]:
        self._sql_generation_metrics.sql_execution_attempts += 1
        try:
            safe_sql = ensure_limit(sql, limit=self.row_limit)
            dataframe = self.connector.run_query(database_path, safe_sql)
        except Exception as exc:
            self._sql_generation_metrics.sql_execution_failures += 1
            return {"ok": False, "error": str(exc), "sql": sql}

        self._sql_generation_metrics.sql_execution_successes += 1
        preview_json = dataframe.head(5).to_json(orient="records", date_format="iso")
        return {
            "ok": True,
            "safe_sql": safe_sql,
            "row_count": len(dataframe),
            "columns": [str(column) for column in dataframe.columns],
            "preview_rows": json.loads(preview_json),
        }

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
