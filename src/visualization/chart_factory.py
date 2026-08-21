"""Chart factory for creating company-style PNG visualizations from DataFrames."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from src.visualization.company_style import CompanyStyle


SUPPORTED_CHART_TYPES = {"auto", "bar", "line", "scatter", "table"}


class ChartCreationError(RuntimeError):
    """Raised when a chart cannot be created from a DataFrame."""


class ChartFactory:
    # Loads the company chart style used by all chart outputs.
    def __init__(self, style: CompanyStyle | None = None) -> None:
        self.style = style or CompanyStyle.from_config()

    # Creates a chart image file from a DataFrame.
    def create_chart(
        self,
        dataframe: pd.DataFrame,
        output_path: str | Path,
        title: str,
        chart_type: str = "auto",
    ) -> tuple[str, Path]:
        if chart_type not in SUPPORTED_CHART_TYPES:
            supported = ", ".join(sorted(SUPPORTED_CHART_TYPES))
            raise ChartCreationError(
                f"Unsupported chart type '{chart_type}'. Supported types: {supported}."
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Forced chart types are respected where possible; invalid data shapes
        # safely fall back to table output instead of crashing the UI.
        selected_type = self.choose_chart_type(dataframe, chart_type)
        figure, axis = plt.subplots(figsize=self.style.figsize, dpi=100)

        try:
            self._apply_base_style(figure, axis)

            try:
                if selected_type == "bar":
                    self._draw_bar(axis, dataframe, title)
                elif selected_type == "line":
                    self._draw_line(axis, dataframe, title)
                elif selected_type == "scatter":
                    self._draw_scatter(axis, dataframe, title)
                else:
                    self._draw_table(axis, dataframe, title)
            except (IndexError, KeyError, ValueError):
                selected_type = "table"
                axis.clear()
                self._apply_base_style(figure, axis)
                self._draw_table(axis, dataframe, title)

            figure.tight_layout()
            figure.savefig(output, bbox_inches="tight", facecolor=self.style.background_color)
        finally:
            plt.close(figure)

        return selected_type, output

    # Recommends the most suitable chart type without considering a user override.
    def recommend_chart_type(self, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return "table"

        numeric_columns = self._numeric_columns(dataframe)
        categorical_columns = self._categorical_columns(dataframe)
        date_columns = self._date_columns(dataframe)

        if date_columns and numeric_columns:
            return "line"

        if categorical_columns and numeric_columns:
            return "bar"

        if len(numeric_columns) >= 2:
            return "scatter"

        if len(numeric_columns) == 1:
            return "line"

        return "table"

    # Chooses the most suitable chart type for the DataFrame shape.
    def choose_chart_type(self, dataframe: pd.DataFrame, requested_type: str = "auto") -> str:
        if dataframe.empty:
            return "table"

        numeric_columns = self._numeric_columns(dataframe)

        if requested_type == "table":
            return "table"

        if requested_type == "auto":
            return self.recommend_chart_type(dataframe)

        if requested_type == "line" and numeric_columns:
            return "line"

        if requested_type == "scatter" and numeric_columns:
            return "scatter"

        if requested_type == "bar" and numeric_columns:
            return "bar"

        return "table"

    # Explains when a forced chart differs from the agent's recommendation.
    def build_recommendation_message(
        self,
        dataframe: pd.DataFrame,
        requested_type: str,
        selected_type: str,
        recommended_type: str,
    ) -> str:
        if requested_type == "auto" or requested_type == recommended_type:
            return ""

        if selected_type != requested_type:
            return (
                f"Requested chart '{requested_type}' could not be drawn safely, "
                f"so the system rendered '{selected_type}'. Recommended chart: "
                f"'{recommended_type}' because {self._recommendation_reason(dataframe, recommended_type)}"
            )

        return (
            f"Requested chart '{requested_type}' was rendered. Recommended chart: "
            f"'{recommended_type}' because {self._recommendation_reason(dataframe, recommended_type)}"
        )

    # Applies shared company styling to a Matplotlib figure and axis.
    def _apply_base_style(self, figure, axis) -> None:
        figure.patch.set_facecolor(self.style.background_color)
        axis.set_facecolor(self.style.background_color)
        plt.rcParams["font.family"] = self.style.font_family
        axis.grid(True, axis="y", color=self.style.grid_color, linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color(self.style.grid_color)

    # Draws a horizontal bar chart for category-and-number results.
    def _draw_bar(self, axis, dataframe: pd.DataFrame, title: str) -> None:
        categorical_columns = self._categorical_columns(dataframe)
        value_column = self._numeric_columns(dataframe)[0]
        columns = [value_column]
        category_column = categorical_columns[0] if categorical_columns else None
        if category_column is not None:
            columns.insert(0, category_column)

        plot_data = dataframe[columns].head(15).iloc[::-1]
        labels = (
            plot_data[category_column].astype(str)
            if category_column is not None
            else [str(index) for index in range(len(plot_data), 0, -1)]
        )

        axis.barh(
            labels,
            plot_data[value_column],
            color=self.style.primary_color,
        )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel(value_column.replace("_", " ").title())
        axis.set_ylabel(
            category_column.replace("_", " ").title()
            if category_column is not None
            else "Row"
        )

    # Draws a line chart for trends or ordered numeric results.
    def _draw_line(self, axis, dataframe: pd.DataFrame, title: str) -> None:
        x_column, value_column = self._choose_xy_columns(dataframe)
        if value_column is None:
            self._draw_table(axis, dataframe, title)
            return

        columns = [value_column]
        if x_column is not None:
            columns.insert(0, x_column)

        plot_data = dataframe[columns].copy().head(100)
        x_values, x_label = self._build_x_values(axis, plot_data, x_column, sort_dates=True)

        if plot_data.empty:
            self._draw_table(axis, dataframe, title)
            return

        axis.plot(
            x_values,
            plot_data[value_column],
            color=self.style.primary_color,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel(x_label)
        axis.set_ylabel(value_column.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=30)

    # Draws a scatter plot for numeric relationships or indexed values.
    def _draw_scatter(self, axis, dataframe: pd.DataFrame, title: str) -> None:
        x_column, y_column = self._choose_xy_columns(dataframe)
        if y_column is None:
            self._draw_table(axis, dataframe, title)
            return

        columns = [y_column]
        if x_column is not None:
            columns.insert(0, x_column)

        plot_data = dataframe[columns].copy().head(200)
        x_values, x_label = self._build_x_values(axis, plot_data, x_column, sort_dates=False)

        axis.scatter(
            x_values,
            plot_data[y_column],
            color=self.style.primary_color,
            edgecolors=self.style.secondary_color,
            alpha=0.75,
        )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_column.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=30)

    # Draws a compact table when charting is not appropriate.
    def _draw_table(self, axis, dataframe: pd.DataFrame, title: str) -> None:
        axis.axis("off")
        axis.set_title(title, loc="left", weight="bold")

        if dataframe.empty:
            axis.text(
                0.5,
                0.5,
                "No rows returned",
                ha="center",
                va="center",
                fontsize=14,
                color=self.style.primary_color,
            )
            return

        table_data = dataframe.head(10).copy()
        table_data = table_data.astype(str)
        table = axis.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor(self.style.grid_color)
            if row == 0:
                cell.set_text_props(weight="bold", color="#FFFFFF")
                cell.set_facecolor(self.style.primary_color)
            else:
                cell.set_facecolor(self.style.background_color)

    # Finds numeric columns that can be used as chart values.
    def _numeric_columns(self, dataframe: pd.DataFrame) -> list[str]:
        return [column for column in dataframe.columns if is_numeric_dtype(dataframe[column])]

    # Finds non-numeric, non-date columns that can label chart categories.
    def _categorical_columns(self, dataframe: pd.DataFrame) -> list[str]:
        return [
            column
            for column in dataframe.columns
            if not is_numeric_dtype(dataframe[column])
            and not self._is_date_like_column(dataframe, column)
        ]

    # Finds columns that appear to contain date or time values.
    def _date_columns(self, dataframe: pd.DataFrame) -> list[str]:
        return [
            column
            for column in dataframe.columns
            if self._is_date_like_column(dataframe, column)
        ]

    # Picks x and y columns for line and scatter charts.
    def _choose_xy_columns(self, dataframe: pd.DataFrame) -> tuple[str | None, str | None]:
        numeric_columns = self._numeric_columns(dataframe)
        if not numeric_columns:
            return None, None

        y_column = numeric_columns[-1]
        date_columns = [column for column in self._date_columns(dataframe) if column != y_column]
        categorical_columns = [
            column for column in self._categorical_columns(dataframe) if column != y_column
        ]
        other_numeric_columns = [column for column in numeric_columns if column != y_column]

        if date_columns:
            return date_columns[0], y_column

        if categorical_columns:
            return categorical_columns[0], y_column

        if other_numeric_columns:
            return other_numeric_columns[0], y_column

        return None, y_column

    # Converts a selected x column into plottable values and an axis label.
    def _build_x_values(
        self,
        axis,
        plot_data: pd.DataFrame,
        x_column: str | None,
        sort_dates: bool,
    ):
        if x_column is None:
            return list(range(1, len(plot_data) + 1)), "Row"

        if self._is_date_like_column(plot_data, x_column):
            plot_data[x_column] = pd.to_datetime(plot_data[x_column], errors="coerce")
            plot_data.dropna(subset=[x_column], inplace=True)
            if sort_dates:
                plot_data.sort_values(x_column, inplace=True)
            return plot_data[x_column], x_column.replace("_", " ").title()

        if is_numeric_dtype(plot_data[x_column]):
            return plot_data[x_column], x_column.replace("_", " ").title()

        positions = list(range(len(plot_data)))
        labels = plot_data[x_column].astype(str).tolist()
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        return positions, x_column.replace("_", " ").title()

    # Detects date-like columns using dtype, name hints, and parse success.
    def _is_date_like_column(self, dataframe: pd.DataFrame, column: str) -> bool:
        if is_datetime64_any_dtype(dataframe[column]):
            return True

        column_name = column.lower()
        if not any(keyword in column_name for keyword in ("date", "year", "month", "time")):
            return False

        parsed = pd.to_datetime(dataframe[column], errors="coerce")
        return parsed.notna().mean() >= 0.7

    # Gives a short reason for the recommended chart type.
    def _recommendation_reason(self, dataframe: pd.DataFrame, recommended_type: str) -> str:
        numeric_count = len(self._numeric_columns(dataframe))
        categorical_count = len(self._categorical_columns(dataframe))
        date_count = len(self._date_columns(dataframe))

        if recommended_type == "line" and date_count:
            return "the result contains a date/time column and numeric values."

        if recommended_type == "line":
            return "the result mainly contains one numeric series."

        if recommended_type == "bar":
            return "the result contains categorical labels and numeric values."

        if recommended_type == "scatter":
            return "the result contains multiple numeric columns."

        if recommended_type == "table":
            return (
                "the result has no clear numeric chart structure "
                f"({categorical_count} categorical columns, {numeric_count} numeric columns)."
            )

        return "it best matches the detected result shape."
