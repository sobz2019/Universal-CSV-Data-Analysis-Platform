import os
import io
import re
import json
import base64
import warnings

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Optional: requests is only needed for fetching GeoJSON from URLs
try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# ──────────────────────────────────────────────────────────────────────────────
# GeoJSON helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_geojson_text(text: str):
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Invalid GeoJSON: {e}")


def get_local_geojson(path: str):
    """Try to load a GeoJSON from disk if present."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_uk_lad_geojson():
    """
    Prefer a local file over the network. Then try multiple public URLs.
    Returns (geojson, source_label). Raises on failure.
    """
    # 1) Local file (drop one into assets/ or data/ to be offline-safe)
    for p in ["assets/uk_lad.geojson", "data/uk_lad.geojson"]:
        gj = get_local_geojson(p)
        if gj:
            return gj, f"local:{p}"

    # 2) Public mirrors (paths may change; we try several)
    if requests is None:
        raise RuntimeError("GeoJSON fetch requires 'requests'. Upload a file or add 'requests' to dependencies.")

    candidates = [
        # Community mirror (contains `name` and `code` in properties)
        "https://raw.githubusercontent.com/ajparsons/geojson-uk/master/json/la.json",
        # Older mirrors kept as fallbacks (may 404)
        "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/lad.json",
        "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/localauthority/lad.json",
    ]

    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            gj = r.json()
            if isinstance(gj, dict) and gj.get("features"):
                return gj, url
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not fetch UK GeoJSON from fallback list. Last error: {last_err}")


def fetch_geojson(url: str):
    if requests is None:
        raise RuntimeError("Custom/UK GeoJSON fetch needs 'requests'. Add it to your dependencies.")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def infer_featureidkey(geojson: dict, id_series: pd.Series):
    """
    Try to find which GeoJSON property best matches the given ID column values.
    Returns a featureidkey like 'properties.LAD23CD' or 'properties.name', or None.
    """
    ids = set(str(v).strip().upper() for v in id_series.dropna().astype(str))
    if not ids:
        return None

    counts = {}
    for feat in geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        for k, v in props.items():
            if v is None:
                continue
            if str(v).strip().upper() in ids:
                counts[k] = counts.get(k, 0) + 1

    if not counts:
        return None
    best_key = max(counts, key=counts.get)
    return f"properties.{best_key}"


def guess_locationmode(series: pd.Series) -> str:
    """Heuristics to pick country names vs ISO-2 vs ISO-3 for world choropleths."""
    vals = series.dropna().astype(str).str.strip()
    up = vals.str.upper()
    is_iso3 = (up.str.len() == 3) & up.str.match(r"^[A-Z]{3}$")
    is_iso2 = (up.str.len() == 2) & up.str.match(r"^[A-Z]{2}$")
    if is_iso3.mean() > 0.8:
        return "ISO-3"
    if is_iso2.mean() > 0.8:
        return "ISO-2"
    return "country names"


# ──────────────────────────────────────────────────────────────────────────────
# App + Data helpers
# ──────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()

    def clean_data(self):
        report = []
        for col in self.df.columns:
            miss = self.df[col].isna().mean() * 100
            if miss > 50:
                report.append(f"Column '{col}' has {miss:.1f}% missing values - consider removing")
            elif miss > 0:
                if self.df[col].dtype in ["int64", "float64"]:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    report.append(f"Filled {miss:.1f}% missing in '{col}' with median")
                else:
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                        report.append(f"Filled {miss:.1f}% missing in '{col}' with mode")
        dups = self.df.duplicated().sum()
        if dups > 0:
            self.df.drop_duplicates(inplace=True)
            report.append(f"Removed {dups} duplicate rows")
        if not report:
            report.append("No cleaning required")
        return report

    def generate_insights(self):
        ins = [f"Dataset contains {len(self.df)} rows and {len(self.df.columns)} columns"]
        missing_cols = self.df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        ins.append("No missing values detected" if len(missing_cols) == 0 else f"{len(missing_cols)} columns have missing values")
        num = self.df.select_dtypes(include=[np.number]).columns
        cat = self.df.select_dtypes(include=["object"]).columns
        if len(num) > 0:
            ins.append(f"Found {len(num)} numeric columns for analysis")
        if len(cat) > 0:
            ins.append(f"Found {len(cat)} categorical columns")
        return ins

    def statistical_analysis(self):
        results = {"correlations": [], "outliers": {}, "summary_stats": {}}
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            corr = self.df[num_cols].corr()
            for i, c1 in enumerate(num_cols):
                for j, c2 in enumerate(num_cols):
                    if i < j:
                        v = corr.loc[c1, c2]
                        if not np.isnan(v) and abs(v) > 0.5:
                            results["correlations"].append(
                                {"var1": c1, "var2": c2, "correlation": round(v, 3),
                                 "strength": "Strong" if abs(v) > 0.7 else "Moderate"}
                            )
        for col in num_cols:
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
            results["summary_stats"][col] = {
                "mean": round(data.mean(), 2),
                "median": round(data.median(), 2),
                "std": round(data.std(), 2),
                "min": round(data.min(), 2),
                "max": round(data.max(), 2),
            }
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            outs = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            results["outliers"][col] = {"count": len(outs), "percentage": round(len(outs) / len(data) * 100, 1)}
        return results

    def business_insights(self):
        out = {"recommendations": [], "kpis": {}}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            d = self.df[col].dropna()
            if len(d) == 0:
                continue
            out["kpis"][col] = {
                "total": float(d.sum()),
                "average": float(d.mean()),
                "growth_potential": "High" if d.std() / (d.mean() if d.mean() else 1) > 0.5 else "Low",
            }
        out["recommendations"] = [
            "📊 Review high-variance columns for data quality issues",
            "🔍 Investigate correlations between key variables",
            "📈 Segment analysis by categorical variables",
            "⚠️ Address outliers that may skew analysis results",
        ]
        return out


def parse_uploaded_file(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename.lower():
            for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode(enc)))
                    return df, f"Loaded {filename} ({len(df)} rows, {len(df.columns)} cols)"
                except UnicodeDecodeError:
                    continue
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="ignore")))
            return df, f"Loaded {filename} (best-effort encoding)"
        return None, "Please upload a CSV file"
    except Exception as e:
        return None, f"Error loading file: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# UI builders
# ──────────────────────────────────────────────────────────────────────────────
def create_complete_dashboard(df, analysis):
    return [
        dcc.Tabs(
            id="nav-tabs",
            value="overview",
            children=[
                dcc.Tab(label="📊 Overview", value="overview"),
                dcc.Tab(label="📈 Charts", value="charts"),
                dcc.Tab(label="🗺️ Maps", value="maps"),
                dcc.Tab(label="📊 Statistics", value="stats"),
                dcc.Tab(label="💡 Insights", value="insights"),
                dcc.Tab(label="📋 Data", value="data"),
            ],
            style={"marginBottom": 20},
        ),
        html.Div([
            html.Div(id="overview-content",  children=create_overview_content(df, analysis)),
            html.Div(id="charts-content",    children=create_charts_content(df, analysis),  style={"display": "none"}),
            html.Div(id="maps-content",      children=create_maps_content(df, analysis),    style={"display": "none"}),
            html.Div(id="stats-content",     children=create_statistics_content(df, analysis), style={"display": "none"}),
            html.Div(id="insights-content",  children=create_insights_content(df, analysis), style={"display": "none"}),
            html.Div(id="data-content",      children=create_data_content(df),              style={"display": "none"}),
        ])
    ]


def create_overview_content(df, analysis):
    return [
        html.H3("📊 Dataset Overview", style={"color": "#2c3e50"}),
        html.Div([
            html.Div([html.H2(f"{len(df)}", style={"color":"#e74c3c","margin":0,"textAlign":"center"}),
                      html.P("Total Rows", style={"margin":0,"textAlign":"center"})], className="summary-card"),
            html.Div([html.H2(f"{len(df.columns)}", style={"color":"#3498db","margin":0,"textAlign":"center"}),
                      html.P("Total Columns", style={"margin":0,"textAlign":"center"})], className="summary-card"),
            html.Div([html.H2(f"{df.isnull().sum().sum()}", style={"color":"#f39c12","margin":0,"textAlign":"center"}),
                      html.P("Missing Values", style={"margin":0,"textAlign":"center"})], className="summary-card"),
            html.Div([html.H2(f"{len(analysis['numeric_cols'])}", style={"color":"#27ae60","margin":0,"textAlign":"center"}),
                      html.P("Numeric Columns", style={"margin":0,"textAlign":"center"})], className="summary-card"),
        ], className="summary-row"),
        html.H3("🧹 Data Cleaning Report", style={"color":"#2c3e50","marginTop":30}),
        html.Div([html.Ul([html.Li(i) for i in analysis["cleaning_report"]])], className="report-box"),
        html.H3("💡 Quick Insights", style={"color":"#2c3e50","marginTop":30}),
        html.Div([html.Ul([html.Li(x) for x in analysis["insights"]])], className="report-box"),
        html.H3("📋 Column Details", style={"color":"#2c3e50","marginTop":30}),
        dash_table.DataTable(
            data=[{"Column":c,"Type":str(df[c].dtype),"Unique":df[c].nunique(),
                   "Missing":df[c].isnull().sum(),
                   "Sample":str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else "N/A"} for c in df.columns],
            columns=[{"name":n,"id":n} for n in ["Column","Type","Unique","Missing","Sample"]],
            style_cell={"textAlign":"left","padding":"10px"},
            style_header={"backgroundColor":"#3498db","color":"white","fontWeight":"bold"},
            page_size=15
        )
    ]


def create_charts_content(df, analysis):
    numeric_cols = analysis["numeric_cols"]
    all_cols = list(df.columns)

    return [
        html.H3("📈 Interactive Data Visualizations", style={"color": "#2c3e50"}),

        html.Div([
            html.Div([
                html.Label("Chart Type:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="chart-type-selector",
                    options=[{"label": t, "value": v} for t, v in [
                        ("Bar Chart", "bar"),
                        ("Histogram", "histogram"),
                        ("Box Plot", "box"),
                        ("Scatter Plot", "scatter"),
                        ("Line Chart", "line"),
                        ("Correlation Heatmap", "heatmap"),
                    ]],
                    value="bar",
                ),
            ], style={"width": "30%", "display": "inline-block", "marginRight": "5%"}),

            html.Div([
                html.Label("X-Axis Column:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="chart-x-axis",
                    options=[{"label": c, "value": c} for c in all_cols],
                    value=all_cols[0] if all_cols else None,
                ),
            ], style={"width": "30%", "display": "inline-block", "marginRight": "5%"}),

            html.Div([
                html.Label("Y-Axis Column:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="chart-y-axis",
                    options=[{"label": c, "value": c} for c in all_cols],
                    value=(numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else None)),
                ),
            ], style={"width": "30%", "display": "inline-block"}),
        ], style={"marginBottom": 20}),

        html.Div(
            id="chart-recommendations",
            style={"marginBottom": 15, "padding": 10, "backgroundColor": "#e8f4f8", "borderRadius": 5},
        ),
        dcc.Graph(id="dynamic-chart", style={"height": 600}),
    ]


def create_maps_content(df, analysis):
    numeric_cols = analysis["numeric_cols"]
    all_cols = list(df.columns)

    return [
        html.H3("🗺️ Geographic Analysis", style={"color":"#2c3e50"}),
        html.P("Create choropleth heat maps for countries/US states, UK local authorities, your own GeoJSON regions, or scatter maps from latitude/longitude.",
               style={"marginBottom":20,"fontStyle":"italic"}),

        # High-level map type + geography scope
        html.Div([
            html.Div([
                html.Label("Map Type:", style={"fontWeight":"bold"}),
                dcc.RadioItems(
                    id="map-type-selector",
                    options=[
                        {"label":" Choropleth Heat Map", "value":"choropleth"},
                        {"label":" Regional Bar Chart", "value":"bar"},
                        {"label":" Scatter Map Points", "value":"scatter"},
                    ],
                    value="choropleth", inline=True
                ),
            ], style={"width":"48%", "display":"inline-block", "verticalAlign":"top"}),

            html.Div([
                html.Label("Geography:", style={"fontWeight":"bold"}),
                dcc.Dropdown(
                    id="geo-scope",
                    options=[
                        {"label": "World (countries)", "value": "world"},
                        {"label": "USA (states)", "value": "usa"},
                        {"label": "UK (Local Authorities — auto)", "value": "uk_auto"},
                        {"label": "Custom GeoJSON", "value": "custom"},
                    ],
                    value="world",
                    clearable=False,
                ),
            ], style={"width":"48%", "display":"inline-block", "marginLeft":"4%"}),
        ], style={"marginBottom":15}),

        # Core selectors (always visible)
        html.Div([
            html.Div([
                html.Label("Location/Region Column:", style={"fontWeight":"bold", "marginBottom":"5px"}),
                dcc.Dropdown(id="map-location-col",
                             options=[{"label":c,"value":c} for c in all_cols],
                             value=all_cols[0] if all_cols else None),
            ], style={"width":"30%","display":"inline-block","marginRight":"5%"}),

            html.Div([
                html.Label("Value Column (for coloring/sizing):", style={"fontWeight":"bold", "marginBottom":"5px"}),
                dcc.Dropdown(id="map-value-col",
                             options=[{"label":c,"value":c} for c in numeric_cols],
                             value=numeric_cols[0] if numeric_cols else None),
            ], style={"width":"30%","display":"inline-block","marginRight":"5%"}),

            html.Div([
                html.Label("Latitude (for Scatter):", style={"fontWeight":"bold","marginBottom":"5px"}),
                dcc.Dropdown(id="map-lat-col", options=[{"label":c,"value":c} for c in all_cols], value=None, disabled=True),
            ], style={"width":"16%","display":"inline-block","marginRight":"2%"}),

            html.Div([
                html.Label("Longitude (for Scatter):", style={"fontWeight":"bold","marginBottom":"5px"}),
                dcc.Dropdown(id="map-lon-col", options=[{"label":c,"value":c} for c in all_cols], value=None, disabled=True),
            ], style={"width":"16%","display":"inline-block"}),
        ], style={"marginBottom":10}),

        # Custom/UK overrides: upload or URL + featureid
        html.Div(id="custom-geo-controls", children=[
            html.Div([
                html.Label("GeoJSON URL:", style={"fontWeight":"bold"}),
                dcc.Input(id="geojson-url", type="text", placeholder="https://.../your_regions.geojson", style={"width":"100%"}),
            ], style={"width":"38%","display":"inline-block","marginRight":"2%"}),

            html.Div([
                html.Label("Feature ID property (featureidkey):", style={"fontWeight":"bold"}),
                dcc.Input(id="geojson-featureid", type="text", value="",  # leave empty to auto-detect
                          placeholder="e.g., properties.name or properties.LAD23CD", style={"width":"100%"}),
            ], style={"width":"30%","display":"inline-block","marginRight":"2%"}),

            html.Div([
                html.Label("Your column that matches the feature ID:", style={"fontWeight":"bold"}),
                dcc.Dropdown(id="custom-id-col", options=[{"label":c,"value":c} for c in all_cols],
                             value=all_cols[0] if all_cols else None),
            ], style={"width":"28%","display":"inline-block"}),

            html.Div([
                html.Label("Or upload a GeoJSON file:", style={"fontWeight":"bold", "display":"block", "marginTop":"8px"}),
                dcc.Upload(
                    id="geojson-upload",
                    children=html.Div(["📄 Drag & drop or click to select a .geojson file"]),
                    multiple=False,
                    style={
                        "width":"100%","height":"58px","lineHeight":"58px","borderWidth":"1px",
                        "borderStyle":"dashed","borderRadius":"6px","textAlign":"center",
                        "backgroundColor":"#fafafa","cursor":"pointer"
                    },
                ),
                html.Small(
                    "Tip: place a file at assets/uk_lad.geojson to load automatically without network.",
                    style={"color":"#6c757d"}
                )
            ], style={"width":"100%","marginTop":"6px"}),
        ], style={"marginBottom":15, "backgroundColor":"#f8f9fa", "padding":"10px", "borderRadius":"8px", "display":"none"}),

        dcc.Graph(id="interactive-map", style={"height":600}),
    ]


def create_statistics_content(df, analysis):
    stats_results = analysis.get("statistical_results", {})
    content = [html.H3("📊 Statistical Analysis", style={"color":"#2c3e50"})]
    correlations = stats_results.get("correlations", [])
    if correlations:
        content += [html.H4("🔗 Significant Correlations", style={"color":"#34495e"}),
                    dash_table.DataTable(
                        data=correlations,
                        columns=[{"name":"Variable 1","id":"var1"},{"name":"Variable 2","id":"var2"},
                                 {"name":"Correlation","id":"correlation"},{"name":"Strength","id":"strength"}],
                        style_header={"backgroundColor":"#3498db","color":"white"},
                        style_cell={"textAlign":"left","padding":"10px"})]
    summary_stats = stats_results.get("summary_stats", {})
    if summary_stats:
        rows = [{"Variable":c,"Mean":s["mean"],"Median":s["median"],"Std Dev":s["std"],"Min":s["min"],"Max":s["max"]}
                for c,s in summary_stats.items()]
        content += [html.H4("📈 Summary Statistics", style={"color":"#34495e","marginTop":30}),
                    dash_table.DataTable(
                        data=rows, columns=[{"name":k,"id":k} for k in rows[0].keys()],
                        style_header={"backgroundColor":"#27ae60","color":"white"},
                        style_cell={"textAlign":"left","padding":"10px"})]
    outliers = stats_results.get("outliers", {})
    if outliers:
        rows = [{"Variable":c,"Outlier Count":v["count"],"Outlier %":v["percentage"]} for c,v in outliers.items()]
        content += [html.H4("⚠️ Outlier Analysis", style={"color":"#34495e","marginTop":30}),
                    dash_table.DataTable(
                        data=rows, columns=[{"name":k,"id":k} for k in rows[0].keys()],
                        style_header={"backgroundColor":"#e74c3c","color":"white"},
                        style_cell={"textAlign":"left","padding":"10px"})]
    if not correlations and not summary_stats:
        content.append(html.P("Statistical analysis requires numeric data columns.", style={"textAlign":"center","padding":20}))
    return content


def create_insights_content(df, analysis):
    biz = analysis.get("business_insights", {})
    content = [html.H3("💡 Business Insights & KPIs", style={"color":"#2c3e50"})]
    kpis = biz.get("kpis", {})
    if kpis:
        cards = []
        for col, k in list(kpis.items())[:4]:
            cards.append(html.Div([
                html.H5(col, style={"margin":0,"color":"white"}),
                html.H3(f"{k['total']:,.0f}", style={"margin":0,"color":"white"}),
                html.P(f"Average: {k['average']:.1f}", style={"margin":0,"color":"rgba(255,255,255,0.8)"}),
                html.P(f"Growth Potential: {k['growth_potential']}", style={"margin":0,"fontSize":12,"color":"rgba(255,255,255,0.8)"}),
            ], className="kpi-card"))
        content += [html.H4("📊 Key Performance Indicators", style={"color":"#34495e"}), html.Div(cards, className="kpi-row")]
    recs = biz.get("recommendations", [])
    if recs:
        content += [html.H4("💡 Strategic Recommendations", style={"color":"#34495e","marginTop":30}),
                    html.Div([html.Ul([html.Li(r, style={"marginBottom":8}) for r in recs])], className="report-box")]
    if not kpis and not recs:
        content.append(html.P("Business insights will be generated based on your data structure.", style={"textAlign":"center","padding":20}))
    return content


def create_data_content(df):
    return [
        html.H3("📋 Data Explorer", style={"color":"#2c3e50"}),
        html.P(f"Displaying first 100 rows of {len(df)} total rows", style={"fontStyle":"italic","marginBottom":15}),
        dash_table.DataTable(
            data=df.head(100).to_dict("records"),
            columns=[{"name":c,"id":c} for c in df.columns],
            style_cell={"textAlign":"left","padding":"8px","maxWidth":"150px","overflow":"hidden","textOverflow":"ellipsis"},
            style_header={"backgroundColor":"#3498db","color":"white","fontWeight":"bold"},
            page_size=25, sort_action="native", filter_action="native", style_table={"overflowX":"auto"})
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="dataset"),
    dcc.Store(id="analysis-results"),
    dcc.Store(id="geojson-store"),  # holds uploaded geojson (and tiny error messages)

    html.H1("🔍 Universal CSV Data Analysis Platform", style={"textAlign":"center","color":"#2c3e50","marginBottom":10}),
    html.P("Upload any CSV file for automatic cleaning, analysis, and visualization",
           style={"textAlign":"center","color":"#7f8c8d","fontSize":16,"marginBottom":30}),
    dcc.Upload(id="upload-csv",
               children=html.Div(["📁 Drag and Drop or Click to Select CSV File"], style={"fontSize":18}),
               style={"width":"100%","height":"100px","lineHeight":"100px","borderWidth":"2px","borderStyle":"dashed",
                      "borderRadius":"10px","textAlign":"center","margin":"10px","borderColor":"#3498db",
                      "backgroundColor":"#ecf0f1","cursor":"pointer"},
               multiple=False),
    html.Div(id="status-message", style={"textAlign":"center","marginTop":10}),
    html.Div(id="dashboard", style={"display":"none"})
], style={"padding":20,"fontFamily":"Arial, sans-serif"})


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    [Output("dataset","data"),
     Output("analysis-results","data"),
     Output("status-message","children"),
     Output("dashboard","style"),
     Output("dashboard","children")],
    Input("upload-csv","contents"),
    State("upload-csv","filename")
)
def handle_file_upload(contents, filename):
    if contents is None:
        return {}, {}, "", {"display":"none"}, []
    df, status = parse_uploaded_file(contents, filename)
    if df is None:
        return {}, {}, html.Div(status, style={"color":"red"}), {"display":"none"}, []

    analyzer = DataAnalyzer(df)
    analysis_data = {
        "cleaning_report": analyzer.clean_data(),
        "insights": analyzer.generate_insights(),
        "numeric_cols": list(analyzer.df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(analyzer.df.select_dtypes(include=["object"]).columns),
        "statistical_results": analyzer.statistical_analysis(),
        "business_insights": analyzer.business_insights()
    }
    layout = create_complete_dashboard(analyzer.df, analysis_data)
    msg = html.Div("✅ " + status, style={"color":"green","fontWeight":"bold"})
    return analyzer.df.to_dict("records"), analysis_data, msg, {"display":"block"}, layout


@app.callback(
    [Output("overview-content","style"),
     Output("charts-content","style"),
     Output("maps-content","style"),
     Output("stats-content","style"),
     Output("insights-content","style"),
     Output("data-content","style")],
    Input("nav-tabs","value")
)
def switch_tabs(active):
    styles = [{"display":"none"}]*6
    idx = {"overview":0,"charts":1,"maps":2,"stats":3,"insights":4,"data":5}.get(active,0)
    styles[idx] = {"display":"block"}
    return styles


# Charts
@app.callback(
    Output("dynamic-chart","figure"),
    [Input("chart-type-selector","value"),
     Input("chart-x-axis","value"),
     Input("chart-y-axis","value"),
     Input("dataset","data")]
)
def update_dynamic_chart(chart_type, x_col, y_col, data):
    if not data or not x_col:
        return go.Figure().add_annotation(text="Upload a CSV, then pick X/Y.", showarrow=False, x=0.5, y=0.5)
    df = pd.DataFrame(data)
    try:
        if chart_type == "histogram":
            return px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
        if chart_type == "box":
            return px.box(df, x=x_col, y=y_col) if y_col and y_col != x_col else px.box(df, y=x_col)
        if chart_type == "scatter" and y_col and y_col != x_col:
            return px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        if chart_type == "bar":
            if df[x_col].dtype == "object" and y_col and y_col != x_col:
                tmp = df.groupby(x_col)[y_col].sum(numeric_only=True).reset_index().sort_values(y_col, ascending=False).head(20)
                fig = px.bar(tmp, x=x_col, y=y_col, title=f"Total {y_col} by {x_col}")
                fig.update_xaxes(tickangle=45); return fig
            if df[x_col].dtype == "object":
                vc = df[x_col].value_counts().head(20)
                fig = px.bar(x=vc.index, y=vc.values, labels={"x":x_col,"y":"Count"}, title=f"Count of {x_col}")
                fig.update_xaxes(tickangle=45); return fig
            return px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
        if chart_type == "line" and y_col and y_col != x_col:
            if df[x_col].dtype == "object":
                tmp = df.groupby(x_col)[y_col].sum(numeric_only=True).reset_index().sort_values(y_col)
                fig = px.line(tmp, x=x_col, y=y_col, title=f"Total {y_col} by {x_col}")
                fig.update_xaxes(tickangle=45); return fig
            return px.line(df.sort_values(x_col), x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        if chart_type == "heatmap":
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] > 1:
                return px.imshow(num.corr(), text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale="RdBu")
            return go.Figure().add_annotation(text="Need ≥2 numeric columns for heatmap", showarrow=False, x=0.5, y=0.5)
        return px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
    except Exception as e:
        return go.Figure().add_annotation(text=f"Chart error: {e}", showarrow=False, x=0.5, y=0.5)


@app.callback(
    Output("chart-recommendations","children"),
    [Input("chart-x-axis","value"),
     Input("chart-y-axis","value"),
     Input("dataset","data")]
)
def chart_recs(x_col, y_col, data):
    if not data or not x_col:
        return ""
    df = pd.DataFrame(data)
    recs = []
    x_cat = df[x_col].dtype == "object"
    y_num = y_col and (df[y_col].dtype in ["int64","float64"])
    if x_cat and y_num:
        recs += ["💡 Bar chart is ideal: numeric by category.", "⚠️ Line chart is rarely meaningful with categorical X."]
    elif (not x_cat) and y_num:
        recs.append("💡 Scatter/line works well for numeric↔numeric.")
    elif x_cat and not y_col:
        recs.append("💡 Bar chart of category counts.")
    return html.Div([html.P("📊 Suggestions:", style={"fontWeight":"bold","margin":0}),
                     html.Ul([html.Li(r) for r in recs], style={"margin":5})])


# GeoJSON upload → store
@app.callback(
    Output("geojson-store", "data"),
    Input("geojson-upload", "contents"),
    State("geojson-upload", "filename"),
    prevent_initial_call=True
)
def load_geojson_from_upload(contents, filename):
    if not contents:
        return dash.no_update
    try:
        header, encoded = contents.split(",", 1)
        gj_text = base64.b64decode(encoded).decode("utf-8", errors="ignore")
        gj = parse_geojson_text(gj_text)
        return {"source": f"upload:{filename}", "geojson": gj}
    except Exception as e:
        return {"error": f"GeoJSON upload error: {e}"}


# Maps — UI state toggles
@app.callback(
    [Output("map-lat-col","disabled"),
     Output("map-lon-col","disabled"),
     Output("custom-geo-controls","style")],
    [Input("map-type-selector","value"),
     Input("geo-scope","value")]
)
def toggle_map_controls(map_type, geo_scope):
    is_scatter = (map_type == "scatter")
    custom_style = {
        "marginBottom": 15,
        "backgroundColor": "#f8f9fa",
        "padding": "10px",
        "borderRadius": "8px",
        # Show the custom block for both 'custom' and 'uk_auto' so users can upload/override
        "display": "block" if geo_scope in ("custom", "uk_auto") else "none",
    }
    return (not is_scatter), (not is_scatter), custom_style


# Maps — figure
@app.callback(
    Output("interactive-map","figure"),
    [Input("map-type-selector","value"),
     Input("geo-scope","value"),
     Input("map-location-col","value"),
     Input("map-value-col","value"),
     Input("map-lat-col","value"),
     Input("map-lon-col","value"),
     Input("geojson-url","value"),
     Input("geojson-featureid","value"),
     Input("custom-id-col","value"),
     Input("geojson-store","data"),
     Input("dataset","data")]
)
def update_interactive_map(map_type, geo_scope, loc_col, val_col, lat_col, lon_col,
                           geojson_url, featureidkey, custom_id_col, geojson_store, data):
    if not data:
        return go.Figure().add_annotation(text="Upload data to map.", showarrow=False, x=0.5, y=0.5)

    df = pd.DataFrame(data)

    # Coerce numeric if selected
    if val_col is not None and val_col in df.columns:
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    try:
        # ── Scatter map: requires lat & lon ────────────────────────────────────
        if map_type == "scatter" and lat_col and lon_col:
            return px.scatter_mapbox(
                df.dropna(subset=[lat_col, lon_col]),
                lat=lat_col, lon=lon_col,
                color=val_col if val_col else None,
                size=val_col if val_col else None,
                hover_data=[loc_col] if loc_col else None,
                mapbox_style="open-street-map", zoom=2, height=600,
                title="Geographic Distribution"
            )

        # ── Choropleth modes ───────────────────────────────────────────────────
        if map_type == "choropleth" and loc_col and val_col:
            # Aggregate per region
            agg = df.groupby(loc_col, dropna=True)[val_col].sum(min_count=1).reset_index()
            if agg[val_col].notna().sum() == 0:
                return go.Figure().add_annotation(text="Value column must be numeric for choropleth.", showarrow=False, x=0.5, y=0.5)

            # WORLD countries
            if geo_scope == "world":
                mode = guess_locationmode(agg[loc_col])
                fig = px.choropleth(
                    agg, locations=loc_col, color=val_col,
                    locationmode=mode, hover_name=loc_col,
                    color_continuous_scale="Blues", title=f"{val_col} by {loc_col}"
                )
                fig.update_geos(showframe=False, showcoastlines=True, projection_type="equirectangular")
                fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
                return fig

            # USA states
            if geo_scope == "usa":
                fig = px.choropleth(
                    agg, locations=loc_col, color=val_col,
                    locationmode="USA-states", scope="usa",
                    color_continuous_scale="Blues", title=f"{val_col} by state"
                )
                fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
                return fig

            # UK Local Authorities — auto (prefer uploaded/local; fallback to URLs)
            if geo_scope == "uk_auto":
                # Prefer uploaded file if present
                if isinstance(geojson_store, dict) and geojson_store.get("geojson"):
                    gj = geojson_store["geojson"]
                    used_source = geojson_store.get("source", "upload")
                else:
                    try:
                        gj, used_source = get_uk_lad_geojson()
                    except Exception as e:
                        msg = ("UK auto mode: no uploaded/local GeoJSON and all network fallbacks failed.<br>"
                               "Upload a GeoJSON above or place one at <code>assets/uk_lad.geojson</code>.")
                        return go.Figure().add_annotation(text=f"{msg}<br><br>Detail: {e}",
                                                          showarrow=False, x=0.5, y=0.5)

                fid = infer_featureidkey(gj, agg[loc_col]) or "properties.name"

                fig = px.choropleth(
                    agg, geojson=gj, locations=loc_col, color=val_col,
                    featureidkey=fid, color_continuous_scale="Blues",
                    hover_name=loc_col, title=f"{val_col} by {loc_col}"
                )
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

                # Diagnostics: match count + source
                try:
                    prop_key = fid.split(".", 1)[1]
                    gj_vals = {
                        str(f["properties"][prop_key]).strip().upper()
                        for f in gj.get("features", []) if f.get("properties") and f["properties"].get(prop_key) is not None
                    }
                    matches = sum(
                        1 for v in agg[loc_col].astype(str).str.strip().str.upper()
                        if v in gj_vals
                    )
                    fig.add_annotation(
                        text=f"Matched {matches}/{len(agg)} regions • featureidkey={fid} • source={used_source}",
                        xref="paper", yref="paper", x=0.01, y=0.01, showarrow=False, font=dict(size=10)
                    )
                except Exception:
                    pass

                return fig

            # CUSTOM GeoJSON: uploaded → URL → instruct
            if geo_scope == "custom":
                if isinstance(geojson_store, dict) and geojson_store.get("geojson"):
                    gj = geojson_store["geojson"]
                    used_source = geojson_store.get("source", "upload")
                elif geojson_url:
                    if requests is None:
                        return go.Figure().add_annotation(
                            text="Custom GeoJSON needs 'requests'. Add it to dependencies or upload a file.",
                            showarrow=False, x=0.5, y=0.5
                        )
                    try:
                        gj = fetch_geojson(geojson_url)
                        used_source = geojson_url
                    except Exception as e:
                        return go.Figure().add_annotation(text=f"Could not fetch GeoJSON: {e}", showarrow=False, x=0.5, y=0.5)
                else:
                    return go.Figure().add_annotation(
                        text="Custom mode: upload a GeoJSON or provide a URL.", showarrow=False, x=0.5, y=0.5
                    )

                id_col = custom_id_col if custom_id_col else loc_col
                fid = (featureidkey.strip() if featureidkey else None) or infer_featureidkey(gj, agg[id_col]) or "properties.name"

                fig = px.choropleth(
                    agg, geojson=gj, locations=id_col, color=val_col,
                    featureidkey=fid, color_continuous_scale="Blues",
                    hover_name=id_col, title=f"{val_col} by {id_col}"
                )
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

                # Diagnostics
                try:
                    prop_key = fid.split(".", 1)[1]
                    gj_vals = {
                        str(f["properties"][prop_key]).strip().upper()
                        for f in gj.get("features", []) if f.get("properties") and f["properties"].get(prop_key) is not None
                    }
                    matches = sum(1 for v in agg[id_col].astype(str).str.strip().str.upper() if v in gj_vals)
                    fig.add_annotation(
                        text=f"Matched {matches}/{len(agg)} regions • featureidkey={fid} • source={used_source}",
                        xref="paper", yref="paper", x=0.01, y=0.01, showarrow=False, font=dict(size=10)
                    )
                except Exception:
                    pass

                return fig

        # ── Regional Bar fallback ─────────────────────────────────────────────
        if map_type == "bar" and loc_col and val_col:
            tmp = df.groupby(loc_col, dropna=True)[val_col].sum(min_count=1).reset_index() \
                    .sort_values(val_col, ascending=False).head(30)
            fig = px.bar(tmp, x=loc_col, y=val_col, title=f"{val_col} by {loc_col}")
            fig.update_xaxes(tickangle=45)
            return fig

        # Instructional panel if inputs incomplete
        txt = ("For Choropleth (World): region column = country names or ISO-2/ISO-3 codes.<br>"
               "For Choropleth (USA): region column = 2-letter state codes (e.g., CA, NY).<br>"
               "For UK (Local Authorities — auto): choose your UK region column (names or GSS codes), "
               "or upload a UK LAD GeoJSON.<br>"
               "For Custom: upload a GeoJSON or paste a URL; leave Feature ID empty to auto-detect.")
        return go.Figure().add_annotation(text=txt, showarrow=False, x=0.5, y=0.5)
    except Exception as e:
        return go.Figure().add_annotation(text=f"Map error: {e}", showarrow=False, x=0.5, y=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      .summary-row { display:flex; flex-wrap:wrap; gap:15px; margin-bottom:20px; }
      .summary-card { background:#f8f9fa; padding:20px; border-radius:10px; text-align:center; flex:1; min-width:150px; box-shadow:0 2px 4px rgba(0,0,0,0.1); }
      .report-box { background:#f8f9fa; padding:15px; border-radius:8px; border-left:4px solid #3498db; margin-bottom:15px; }
      .kpi-row { display:flex; flex-wrap:wrap; gap:15px; margin-bottom:20px; }
      .kpi-card { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:20px; border-radius:12px; flex:1; min-width:200px; box-shadow:0 4px 8px rgba(0,0,0,0.1); transition: transform .2s; }
      .kpi-card:hover { transform: translateY(-2px); }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=8050)
