import json

import pytest
from datapulse.config import AnalysisConfig
from datapulse.engine import run_analysis, summarize
from datapulse.exporters.html_exporter import export_html
from datapulse.exporters.json_exporter import export_json


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    import polars as pl

    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    p = tmp_path_factory.mktemp("data") / "sample.csv"
    df.write_csv(p)
    return p


def test_run_analysis(sample_csv):
    report = run_analysis(sample_csv)
    assert report.profile is not None
    assert report.profile.rows == 10
    assert report.profile.columns == 3
    assert "pearson" in report.profile.correlations


def test_summarize(sample_csv):
    report = run_analysis(sample_csv)
    s = summarize(report)
    assert s["rows"] == 10
    assert s["quality_score"] == 1.0


def test_export_json(sample_csv, tmp_path):
    report = run_analysis(sample_csv)
    out = tmp_path / "r.json"
    export_json(report, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["profile"]["rows"] == 10


def test_export_html(sample_csv, tmp_path):
    report = run_analysis(sample_csv)
    out = tmp_path / "r.html"
    export_html(report, out)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "datapulse report" in html
    assert "rows" in html


def test_config_yaml_roundtrip(tmp_path):
    cfg = AnalysisConfig()
    p = tmp_path / "cfg.yaml"
    cfg.dump_yaml(p)
    loaded = AnalysisConfig.from_yaml(p)
    assert loaded == cfg


def test_config_overrides():
    cfg = AnalysisConfig()
    cfg2 = cfg.apply_overrides({"automl.time_budget": 300, "visualization.theme": "dark"})
    assert cfg2.automl.time_budget == 300
    assert cfg2.visualization.theme == "dark"
    assert cfg.automl.time_budget == 1800
