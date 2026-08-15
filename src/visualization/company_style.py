"""Company chart style configuration loaded from config/style.yaml."""

from dataclasses import dataclass

from src.utils.config_loader import load_config


@dataclass(frozen=True)
class CompanyStyle:
    primary_color: str = "#1F77B4"
    secondary_color: str = "#FF7F0E"
    background_color: str = "#FFFFFF"
    grid_color: str = "#D9E2EC"
    font_family: str = "Arial"
    chart_width: int = 900
    chart_height: int = 500

    @classmethod
    def from_config(cls) -> "CompanyStyle":
        config = load_config("style.yaml")
        style_config = config.get("company_style", {})
        return cls(
            primary_color=style_config.get("primary_color", cls.primary_color),
            secondary_color=style_config.get("secondary_color", cls.secondary_color),
            background_color=style_config.get("background_color", cls.background_color),
            grid_color=style_config.get("grid_color", cls.grid_color),
            font_family=style_config.get("font_family", cls.font_family),
            chart_width=int(style_config.get("chart_width", cls.chart_width)),
            chart_height=int(style_config.get("chart_height", cls.chart_height)),
        )

    @property
    def figsize(self) -> tuple[float, float]:
        return self.chart_width / 100, self.chart_height / 100
