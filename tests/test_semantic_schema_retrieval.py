"""Tests for semantic schema metadata indexing and retrieval."""

from src.agents.analysis_agent import DataAnalysisAgent, FALLBACK_SQL_TEMPLATES
from src.db.registry import DatabaseRegistry
from src.retrieval.schema_metadata_index import SemanticSchemaRetriever


# Verifies schema retrieval returns relevant tables, columns, and sample questions.
def test_semantic_schema_retriever_returns_relevant_chinook_context() -> None:
    registry = DatabaseRegistry()
    database_info = registry.get_database("chinook")
    retriever = SemanticSchemaRetriever(top_tables=4, top_columns=12)

    result = retriever.retrieve(
        database_name="chinook",
        database_path=database_info["path"],
        database_description=database_info["description"],
        question="Show total sales by country",
        sample_questions=sorted(FALLBACK_SQL_TEMPLATES["chinook"]),
    )

    assert "Database chinook" in result.schema_text
    assert "Table invoices" in result.schema_text
    assert "invoices" in result.retrieved_tables
    assert any(column in result.retrieved_columns for column in ("invoices.Total", "invoices.BillingCountry"))
    assert any("show total sales by country" in question.lower() for question in result.retrieved_sample_questions)


# Verifies the same metadata index can rank database candidates for routing.
def test_semantic_schema_retriever_can_rank_database_candidates() -> None:
    retriever = SemanticSchemaRetriever(database_registry=DatabaseRegistry())
    sample_questions_by_database = {
        database_name: sorted(templates)
        for database_name, templates in FALLBACK_SQL_TEMPLATES.items()
    }

    candidates = retriever.retrieve_database_candidates(
        question="Which film categories generate the most rental revenue?",
        database_names=["chinook", "northwind", "sakila"],
        sample_questions_by_database=sample_questions_by_database,
        top_k=3,
    )

    assert candidates
    assert candidates[0].database_name == "sakila"
    assert candidates[0].score > 0


# Verifies the Analysis Agent uses retrieved schema context before SQL generation.
def test_analysis_agent_records_retrieved_schema_context() -> None:
    registry = DatabaseRegistry()
    database_info = registry.get_database("chinook")

    # Supplies deterministic SQL so the test focuses on retrieval, not OpenAI.
    def fixed_sql(database_name: str, question: str, schema_text: str) -> str:
        assert "Semantic schema retrieval result" in schema_text
        assert "Table invoices" in schema_text
        return """
        SELECT BillingCountry AS country, ROUND(SUM(Total), 2) AS total_sales
        FROM invoices
        GROUP BY BillingCountry
        ORDER BY total_sales DESC
        """

    agent = DataAnalysisAgent(
        row_limit=5,
        use_openai=False,
        sql_generator=fixed_sql,
    )

    result = agent.analyze(
        database_path=database_info["path"],
        database_name="chinook",
        database_description=database_info["description"],
        question="Show total sales by country",
    )

    assert result.row_count == 5
    assert result.schema_context.startswith("Semantic schema retrieval result")
    assert "invoices" in result.retrieved_tables
    assert any(column.startswith("invoices.") for column in result.retrieved_columns)
