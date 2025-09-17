import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64, io, warnings
warnings.filterwarnings("ignore")

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# ---------- Helpers ----------
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
        if len(num) > 0: ins.append(f"Found {len(num)} numeric columns for analysis")
        if len(cat) > 0: ins.append(f"Found {len(cat)} categorical columns")
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

# ---------- UI builders ----------
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
        html.H3("📊 Dataset Overview", style={"color":"#2c3e50"}),
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
                   "Sample":str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0]>0 else "N/A"} for c in df.columns],
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

        # Controls (3 columns)
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
        ], style={"marginBottom": 20}),  # <-- controls Div CLOSED here

        # Suggestions + chart (siblings of the controls Div)
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
        html.P("Create interactive maps and regional analysis:", style={"marginBottom":20,"fontStyle":"italic"}),
        html.Div([html.Label("Map Type:", style={"fontWeight":"bold","marginBottom":"5px"}),
                  dcc.RadioItems(id="map-type-selector",
                                 options=[{"label":" Choropleth Heat Map","value":"choropleth"},
                                          {"label":" Regional Bar Chart","value":"bar"},
                                          {"label":" Scatter Map Points","value":"scatter"}],
                                 value="choropleth", inline=True, style={"marginBottom":15})]),
        html.Div([
            html.Div([html.Label("Location/Region Column:", style={"fontWeight":"bold","marginBottom":"5px"}),
                      dcc.Dropdown(id="map-location-col",
                                   options=[{"label":c,"value":c} for c in all_cols],
                                   value=all_cols[0] if all_cols else None)],
                     style={"width":"30%","display":"inline-block","marginRight":"5%"}),
            html.Div([html.Label("Value Column (for coloring/sizing):", style={"fontWeight":"bold","marginBottom":"5px"}),
                      dcc.Dropdown(id="map-value-col",
                                   options=[{"label":c,"value":c} for c in numeric_cols],
                                   value=numeric_cols[0] if numeric_cols else None)],
                     style={"width":"30%","display":"inline-block","marginRight":"5%"}),
            html.Div([html.Label("Latitude (for Scatter):", style={"fontWeight":"bold","marginBottom":"5px"}),
                      dcc.Dropdown(id="map-lat-col", options=[{"label":c,"value":c} for c in all_cols], value=None, disabled=True)],
                     style={"width":"17%","display":"inline-block","marginRight":"3%"}),
            html.Div([html.Label("Longitude (for Scatter):", style={"fontWeight":"bold","marginBottom":"5px"}),
                      dcc.Dropdown(id="map-lon-col", options=[{"label":c,"value":c} for c in all_cols], value=None, disabled=True)],
                     style={"width":"17%","display":"inline-block"}),
        ], style={"marginBottom":20}),
        dcc.Graph(id="interactive-map", style={"height":600})
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
                html.P(f"Growth Potential: {k['growth_potential']}", style={"margin":0,"fontSize":12,"color":"rgba(255,255,255,0.8)"})
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

# ---------- Layout ----------
app.layout = html.Div([
    dcc.Store(id="dataset"),
    dcc.Store(id="analysis-results"),
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

# ---------- Callbacks ----------
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

@app.callback(
    [Output("map-lat-col","disabled"),
     Output("map-lon-col","disabled")],
    Input("map-type-selector","value")
)
def enable_scatter_coords(map_type):
    is_scatter = (map_type == "scatter")
    return (not is_scatter), (not is_scatter)

@app.callback(
    Output("interactive-map","figure"),
    [Input("map-type-selector","value"),
     Input("map-location-col","value"),
     Input("map-value-col","value"),
     Input("map-lat-col","value"),
     Input("map-lon-col","value"),
     Input("dataset","data")]
)
def update_interactive_map(map_type, loc_col, val_col, lat_col, lon_col, data):
    if not data:
        return go.Figure().add_annotation(text="Upload data to map.", showarrow=False, x=0.5, y=0.5)
    df = pd.DataFrame(data)

    # Try to coerce value column numeric if present
    if val_col is not None and val_col in df.columns:
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    try:
        if map_type == "choropleth" and loc_col and val_col:
            agg = df.groupby(loc_col, dropna=True)[val_col].sum(min_count=1).reset_index()
            if agg[val_col].notna().sum() == 0:
                return go.Figure().add_annotation(text="Value column must be numeric for choropleth.", showarrow=False, x=0.5, y=0.5)
            for mode in ["country names", "ISO-3", "ISO-2", "USA-states"]:
                try:
                    fig = go.Figure(go.Choropleth(
                        locations=agg[loc_col], z=agg[val_col],
                        locationmode=mode, colorscale="Blues", text=agg[loc_col],
                        colorbar_title=str(val_col),
                        hovertemplate="<b>%{text}</b><br>"+str(val_col)+": %{z:,.0f}<extra></extra>"
                    ))
                    fig.update_layout(title=f"{val_col} by {loc_col}",
                                      geo=dict(showframe=False, showcoastlines=True, projection_type="equirectangular"))
                    return fig
                except Exception:
                    continue
            # Fallback: regional bar chart
            tmp = agg.sort_values(val_col, ascending=False).head(25)
            fig = px.bar(tmp, x=loc_col, y=val_col, title=f"{val_col} by {loc_col} (map fallback)")
            fig.update_xaxes(tickangle=45)
            return fig

        if map_type == "bar" and loc_col and val_col:
            tmp = df.groupby(loc_col, dropna=True)[val_col].sum(min_count=1).reset_index().sort_values(val_col, ascending=False).head(30)
            fig = px.bar(tmp, x=loc_col, y=val_col, title=f"{val_col} by {loc_col}")
            fig.update_xaxes(tickangle=45)
            return fig

        if map_type == "scatter" and lat_col and lon_col:
            return px.scatter_mapbox(
                df.dropna(subset=[lat_col, lon_col]),
                lat=lat_col, lon=lon_col,
                color=val_col if val_col else None,
                size=val_col if val_col else None,
                hover_data=[loc_col] if loc_col else None,
                mapbox_style="open-street-map", zoom=3, height=600,
                title="Geographic Distribution"
            )

        txt = ("For Choropleth: choose a region column (country/state names or ISO codes) and a **numeric** value column.<br>"
               "For Scatter: choose Latitude and Longitude columns.")
        return go.Figure().add_annotation(text=txt, showarrow=False, x=0.5, y=0.5)
    except Exception as e:
        return go.Figure().add_annotation(text=f"Map error: {e}", showarrow=False, x=0.5, y=0.5)

# ---------- Styling ----------
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
