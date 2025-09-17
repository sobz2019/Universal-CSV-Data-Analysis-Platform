import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# Initialize the Dash app with callback exception suppression
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Helper functions for data analysis
class DataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        
    def clean_data(self):
        """Perform automatic data cleaning"""
        cleaning_report = []
        
        # Handle missing values
        for col in self.df.columns:
            missing_pct = self.df[col].isnull().mean() * 100
            
            if missing_pct > 50:
                cleaning_report.append(f"Column '{col}' has {missing_pct:.1f}% missing values - consider removing")
            elif missing_pct > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    cleaning_report.append(f"Filled {missing_pct:.1f}% missing values in '{col}' with median")
                else:
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                        cleaning_report.append(f"Filled {missing_pct:.1f}% missing values in '{col}' with mode")
        
        # Remove duplicates
        duplicates_before = self.df.duplicated().sum()
        if duplicates_before > 0:
            self.df.drop_duplicates(inplace=True)
            cleaning_report.append(f"Removed {duplicates_before} duplicate rows")
        
        if not cleaning_report:
            cleaning_report.append("No cleaning operations needed - data is already clean!")
        
        return cleaning_report
    
    def generate_insights(self):
        """Automatically generate insights from the data"""
        insights = []
        insights.append(f"Dataset contains {len(self.df)} rows and {len(self.df.columns)} columns")
        
        missing_cols = self.df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        if len(missing_cols) > 0:
            insights.append(f"{len(missing_cols)} columns have missing values")
        else:
            insights.append("No missing values detected")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Found {len(numeric_cols)} numeric columns for analysis")
        
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            insights.append(f"Found {len(cat_cols)} categorical columns")
        
        return insights
    
    def statistical_analysis(self):
        """Perform statistical analysis with error handling"""
        results = {
            'correlations': [],
            'outliers': {},
            'summary_stats': {}
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Correlation Analysis
        if len(numeric_cols) > 1:
            try:
                corr_matrix = self.df[numeric_cols].corr()
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:
                            corr_val = corr_matrix.loc[col1, col2]
                            if not np.isnan(corr_val) and abs(corr_val) > 0.5:
                                results['correlations'].append({
                                    'var1': col1,
                                    'var2': col2,
                                    'correlation': round(corr_val, 3),
                                    'strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                                })
            except:
                pass
        
        # Summary Statistics
        for col in numeric_cols:
            try:
                data = self.df[col].dropna()
                if len(data) > 0:
                    results['summary_stats'][col] = {
                        'mean': round(data.mean(), 2),
                        'median': round(data.median(), 2),
                        'std': round(data.std(), 2),
                        'min': round(data.min(), 2),
                        'max': round(data.max(), 2)
                    }
                    
                    # Outlier detection
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
                    results['outliers'][col] = {
                        'count': len(outliers),
                        'percentage': round((len(outliers) / len(data)) * 100, 1)
                    }
            except:
                pass
        
        return results
    
    def business_insights(self):
        """Generate business insights"""
        insights = {
            'recommendations': [],
            'kpis': {}
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Calculate KPIs
        for col in numeric_cols:
            try:
                data = self.df[col].dropna()
                if len(data) > 0:
                    insights['kpis'][col] = {
                        'total': data.sum(),
                        'average': data.mean(),
                        'growth_potential': 'High' if data.std() / data.mean() > 0.5 else 'Low'
                    }
            except:
                pass
        
        # Generate recommendations
        insights['recommendations'] = [
            "📊 Review high-variance columns for data quality issues",
            "🔍 Investigate correlations between key variables",
            "📈 Consider segmenting analysis by categorical variables",
            "⚠️ Address outliers that may skew analysis results"
        ]
        
        return insights

def parse_uploaded_file(contents, filename):
    """Parse uploaded CSV file with error handling"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
                    return df, f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns"
                except UnicodeDecodeError:
                    continue
            
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='ignore')))
            return df, f"Loaded {filename} (some characters may be corrupted due to encoding issues)"
        else:
            return None, "Please upload a CSV file"
        
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

# Complete App Layout - ALL components defined here
app.layout = html.Div([
    # Data storage
    dcc.Store(id='dataset'),
    dcc.Store(id='analysis-results'),
    
    # Header
    html.Div([
        html.H1("🔍 Universal CSV Data Analysis Platform", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.P("Upload any CSV file for automatic cleaning, analysis, and visualization",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16})
    ], style={'marginBottom': 30}),
    
    # Upload section
    html.Div([
        dcc.Upload(
            id='upload-csv',
            children=html.Div([
                '📁 Drag and Drop or Click to Select CSV File'
            ], style={'fontSize': 18}),
            style={
                'width': '100%', 'height': '100px', 'lineHeight': '100px',
                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                'textAlign': 'center', 'margin': '10px', 'borderColor': '#3498db',
                'backgroundColor': '#ecf0f1', 'cursor': 'pointer'
            },
            multiple=False
        ),
        html.Div(id='status-message', style={'textAlign': 'center', 'marginTop': 10})
    ], style={'marginBottom': 30}),
    
    # Main dashboard (initially hidden)
    html.Div(id='dashboard', style={'display': 'none'}, children=[
        # Navigation tabs
        dcc.Tabs(id='dashboard-tabs', value='overview', children=[
            dcc.Tab(label='📊 Overview', value='overview'),
            dcc.Tab(label='📈 Charts', value='charts'),
            dcc.Tab(label='🗺️ Maps', value='maps'),
            dcc.Tab(label='📊 Stats', value='stats'),
            dcc.Tab(label='💡 Insights', value='insights'),
            dcc.Tab(label='📋 Data', value='data')
        ], style={'marginBottom': 20}),
        
        # Content area
        html.Div(id='dashboard-content')
    ])
    
], style={'padding': 20, 'fontFamily': 'Arial, sans-serif'})

# Main upload callback - creates entire dashboard
@app.callback(
    [Output('dataset', 'data'),
     Output('analysis-results', 'data'),
     Output('status-message', 'children'),
     Output('dashboard', 'style'),
     Output('dashboard', 'children')],
    [Input('upload-csv', 'contents')],
    [State('upload-csv', 'filename')]
)
def handle_file_upload(contents, filename):
    if contents is None:
        return {}, {}, "", {'display': 'none'}, []
    
    # Parse file
    df, status_msg = parse_uploaded_file(contents, filename)
    
    if df is None:
        return {}, {}, html.Div(status_msg, style={'color': 'red'}), {'display': 'none'}, []
    
    # Analyze data
    analyzer = DataAnalyzer(df)
    cleaning_report = analyzer.clean_data()
    insights = analyzer.generate_insights()
    
    try:
        statistical_results = analyzer.statistical_analysis()
        business_insights = analyzer.business_insights()
    except:
        statistical_results = {}
        business_insights = {}
    
    # Store analysis results
    analysis_data = {
        'cleaning_report': cleaning_report,
        'insights': insights,
        'numeric_cols': list(analyzer.df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': list(analyzer.df.select_dtypes(include=['object']).columns),
        'statistical_results': statistical_results,
        'business_insights': business_insights
    }
    
    # Create complete dashboard layout
    dashboard_layout = create_complete_dashboard(analyzer.df, analysis_data)
    
    success_msg = html.Div("✅ " + status_msg, style={'color': 'green', 'fontWeight': 'bold'})
    
    return analyzer.df.to_dict('records'), analysis_data, success_msg, {'display': 'block'}, dashboard_layout

def create_complete_dashboard(df, analysis_data):
    """Create the complete dashboard with all components"""
    return [
        # Navigation tabs
        dcc.Tabs(id='nav-tabs', value='overview', children=[
            dcc.Tab(label='📊 Overview', value='overview'),
            dcc.Tab(label='📈 Charts', value='charts'),
            dcc.Tab(label='🗺️ Maps', value='maps'),
            dcc.Tab(label='📊 Statistics', value='stats'),
            dcc.Tab(label='💡 Insights', value='insights'),
            dcc.Tab(label='📋 Data', value='data')
        ], style={'marginBottom': 20}),
        
        # All tab contents (controlled by CSS display)
        html.Div([
            # Overview tab
            html.Div(id='overview-content', children=create_overview_content(df, analysis_data)),
            
            # Charts tab  
            html.Div(id='charts-content', children=create_charts_content(df, analysis_data), style={'display': 'none'}),
            
            # Maps tab
            html.Div(id='maps-content', children=create_maps_content(df, analysis_data), style={'display': 'none'}),
            
            # Statistics tab
            html.Div(id='stats-content', children=create_statistics_content(df, analysis_data), style={'display': 'none'}),
            
            # Insights tab
            html.Div(id='insights-content', children=create_insights_content(df, analysis_data), style={'display': 'none'}),
            
            # Data tab
            html.Div(id='data-content', children=create_data_content(df), style={'display': 'none'})
        ])
    ]

def create_overview_content(df, analysis_data):
    """Create overview tab content"""
    return [
        # Summary cards
        html.H3("📊 Dataset Overview", style={'color': '#2c3e50'}),
        html.Div([
            html.Div([
                html.H2(f"{len(df)}", style={'color': '#e74c3c', 'margin': 0, 'textAlign': 'center'}),
                html.P("Total Rows", style={'margin': 0, 'textAlign': 'center'})
            ], className='summary-card'),
            
            html.Div([
                html.H2(f"{len(df.columns)}", style={'color': '#3498db', 'margin': 0, 'textAlign': 'center'}),
                html.P("Total Columns", style={'margin': 0, 'textAlign': 'center'})
            ], className='summary-card'),
            
            html.Div([
                html.H2(f"{df.isnull().sum().sum()}", style={'color': '#f39c12', 'margin': 0, 'textAlign': 'center'}),
                html.P("Missing Values", style={'margin': 0, 'textAlign': 'center'})
            ], className='summary-card'),
            
            html.Div([
                html.H2(f"{len(analysis_data['numeric_cols'])}", style={'color': '#27ae60', 'margin': 0, 'textAlign': 'center'}),
                html.P("Numeric Columns", style={'margin': 0, 'textAlign': 'center'})
            ], className='summary-card')
        ], className='summary-row'),
        
        # Data cleaning report
        html.H3("🧹 Data Cleaning Report", style={'color': '#2c3e50', 'marginTop': 30}),
        html.Div([
            html.Ul([html.Li(item) for item in analysis_data['cleaning_report']])
        ], className='report-box'),
        
        # Quick insights
        html.H3("💡 Quick Insights", style={'color': '#2c3e50', 'marginTop': 30}),
        html.Div([
            html.Ul([html.Li(insight) for insight in analysis_data['insights']])
        ], className='report-box'),
        
        # Column details table
        html.H3("📋 Column Details", style={'color': '#2c3e50', 'marginTop': 30}),
        dash_table.DataTable(
            data=[{
                'Column': col,
                'Type': str(df[col].dtype),
                'Unique': df[col].nunique(),
                'Missing': df[col].isnull().sum(),
                'Sample': str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A'
            } for col in df.columns],
            columns=[
                {'name': 'Column', 'id': 'Column'},
                {'name': 'Data Type', 'id': 'Type'},
                {'name': 'Unique Values', 'id': 'Unique'},
                {'name': 'Missing Values', 'id': 'Missing'},
                {'name': 'Sample Value', 'id': 'Sample'}
            ],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
            page_size=15
        )
    ]

def create_charts_content(df, analysis_data):
    """Create charts content with interactive controls"""
    numeric_cols = analysis_data['numeric_cols']
    all_cols = list(df.columns)
    
    return [
        html.H3("📈 Interactive Data Visualizations", style={'color': '#2c3e50'}),
        
        # Chart controls
        html.Div([
            html.Div([
                html.Label("Chart Type:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='chart-type-selector',
                    options=[
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Histogram', 'value': 'histogram'},
                        {'label': 'Box Plot', 'value': 'box'},
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Line Chart', 'value': 'line'},
                        {'label': 'Correlation Heatmap', 'value': 'heatmap'}
                    ],
                    value='bar'
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
            
            html.Div([
                html.Label("X-Axis Column:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='chart-x-axis',
                    options=[{'label': col, 'value': col} for col in all_cols],
                    value=all_cols[0] if all_cols else None
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
            
            html.Div([
                html.Label("Y-Axis Column:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='chart-y-axis',
                    options=[{'label': col, 'value': col} for col in all_cols],
                    value=numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else None)
                )
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': 20}),
        
        # Chart recommendations
        html.Div(id='chart-recommendations', style={'marginBottom': 15, 'padding': 10, 'backgroundColor': '#e8f4f8', 'borderRadius': 5}),
        
        # Chart display
        dcc.Graph(id='dynamic-chart', style={'height': 600})
    ]

def create_maps_content(df, analysis_data):
    """Create maps content with proper geographic visualization"""
    numeric_cols = analysis_data['numeric_cols']
    all_cols = list(df.columns)
    
    return [
        html.H3("🗺️ Geographic Analysis", style={'color': '#2c3e50'}),
        
        html.P("Create interactive maps and regional analysis:", style={'marginBottom': 20, 'fontStyle': 'italic'}),
        
        # Map type selector
        html.Div([
            html.Label("Map Type:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.RadioItems(
                id='map-type-selector',
                options=[
                    {'label': ' Choropleth Heat Map', 'value': 'choropleth'},
                    {'label': ' Regional Bar Chart', 'value': 'bar'},
                    {'label': ' Scatter Map Points', 'value': 'scatter'}
                ],
                value='choropleth',
                inline=True,
                style={'marginBottom': 15}
            )
        ]),
        
        # Map controls
        html.Div([
            html.Div([
                html.Label("Location/Region Column:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='map-location-col',
                    options=[{'label': col, 'value': col} for col in all_cols],
                    value=all_cols[0] if all_cols else None
                )
            ], style={'width': '45%', 'display': 'inline-block', 'marginRight': '10%'}),
            
            html.Div([
                html.Label("Value Column (for coloring/sizing):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='map-value-col',
                    options=[{'label': col, 'value': col} for col in numeric_cols],
                    value=numeric_cols[0] if numeric_cols else None
                )
            ], style={'width': '45%', 'display': 'inline-block'})
        ], style={'marginBottom': 20}),
        
        # Additional controls for scatter maps
        html.Div(id='map-extra-controls', style={'marginBottom': 20}),
        
        # Map display
        dcc.Graph(id='interactive-map', style={'height': 600})
    ]

def create_statistics_content(df, analysis_data):
    """Create statistics content"""
    stats_results = analysis_data.get('statistical_results', {})
    
    content = [html.H3("📊 Statistical Analysis", style={'color': '#2c3e50'})]
    
    # Correlation analysis
    correlations = stats_results.get('correlations', [])
    if correlations:
        content.extend([
            html.H4("🔗 Significant Correlations", style={'color': '#34495e'}),
            dash_table.DataTable(
                data=correlations,
                columns=[
                    {'name': 'Variable 1', 'id': 'var1'},
                    {'name': 'Variable 2', 'id': 'var2'},
                    {'name': 'Correlation', 'id': 'correlation'},
                    {'name': 'Strength', 'id': 'strength'}
                ],
                style_header={'backgroundColor': '#3498db', 'color': 'white'},
                style_cell={'textAlign': 'left', 'padding': '10px'}
            )
        ])
    
    # Summary statistics
    summary_stats = stats_results.get('summary_stats', {})
    if summary_stats:
        stats_data = []
        for col, stats in summary_stats.items():
            stats_data.append({
                'Variable': col,
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Std Dev': stats['std'],
                'Min': stats['min'],
                'Max': stats['max']
            })
        
        content.extend([
            html.H4("📈 Summary Statistics", style={'color': '#34495e', 'marginTop': 30}),
            dash_table.DataTable(
                data=stats_data,
                columns=[{'name': col, 'id': col} for col in stats_data[0].keys()] if stats_data else [],
                style_header={'backgroundColor': '#27ae60', 'color': 'white'},
                style_cell={'textAlign': 'left', 'padding': '10px'}
            )
        ])
    
    # Outlier analysis
    outliers = stats_results.get('outliers', {})
    if outliers:
        outlier_data = []
        for col, outlier_info in outliers.items():
            outlier_data.append({
                'Variable': col,
                'Outlier Count': outlier_info['count'],
                'Outlier %': outlier_info['percentage']
            })
        
        content.extend([
            html.H4("⚠️ Outlier Analysis", style={'color': '#34495e', 'marginTop': 30}),
            dash_table.DataTable(
                data=outlier_data,
                columns=[{'name': col, 'id': col} for col in outlier_data[0].keys()] if outlier_data else [],
                style_header={'backgroundColor': '#e74c3c', 'color': 'white'},
                style_cell={'textAlign': 'left', 'padding': '10px'}
            )
        ])
    
    if not correlations and not summary_stats:
        content.append(html.P("Statistical analysis requires numeric data columns.", 
                             style={'textAlign': 'center', 'padding': 20}))
    
    return content

def create_insights_content(df, analysis_data):
    """Create business insights content"""
    business_data = analysis_data.get('business_insights', {})
    
    content = [html.H3("💡 Business Insights & KPIs", style={'color': '#2c3e50'})]
    
    # KPIs
    kpis = business_data.get('kpis', {})
    if kpis:
        kpi_cards = []
        for col, kpi_data in list(kpis.items())[:4]:
            kpi_cards.append(
                html.Div([
                    html.H5(col, style={'margin': 0, 'color': 'white'}),
                    html.H3(f"{kpi_data['total']:,.0f}", style={'margin': 0, 'color': 'white'}),
                    html.P(f"Average: {kpi_data['average']:.1f}", style={'margin': 0, 'color': 'rgba(255,255,255,0.8)'}),
                    html.P(f"Growth Potential: {kpi_data['growth_potential']}", 
                          style={'margin': 0, 'fontSize': 12, 'color': 'rgba(255,255,255,0.8)'})
                ], className='kpi-card')
            )
        
        content.extend([
            html.H4("📊 Key Performance Indicators", style={'color': '#34495e'}),
            html.Div(kpi_cards, className='kpi-row')
        ])
    
    # Recommendations
    recommendations = business_data.get('recommendations', [])
    if recommendations:
        content.extend([
            html.H4("💡 Strategic Recommendations", style={'color': '#34495e', 'marginTop': 30}),
            html.Div([
                html.Ul([html.Li(rec, style={'marginBottom': 8}) for rec in recommendations])
            ], className='report-box')
        ])
    
    if not kpis and not recommendations:
        content.append(html.P("Business insights will be generated based on your data structure.", 
                             style={'textAlign': 'center', 'padding': 20}))
    
    return content

def create_data_content(df):
    """Create data explorer content"""
    return [
        html.H3("📋 Data Explorer", style={'color': '#2c3e50'}),
        html.P(f"Displaying first 100 rows of {len(df)} total rows", 
               style={'fontStyle': 'italic', 'marginBottom': 15}),
        
        dash_table.DataTable(
            data=df.head(100).to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            style_cell={'textAlign': 'left', 'padding': '8px', 'maxWidth': '150px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'},
            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
            page_size=25,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto'}
        )
    ]

# Tab switching callback
@app.callback(
    [Output('overview-content', 'style'),
     Output('charts-content', 'style'),
     Output('maps-content', 'style'),
     Output('stats-content', 'style'),
     Output('insights-content', 'style'),
     Output('data-content', 'style')],
    [Input('nav-tabs', 'value')]
)
def switch_tabs(active_tab):
    # All tabs start hidden
    styles = [{'display': 'none'}] * 6
    
    # Show the active tab
    if active_tab == 'overview':
        styles[0] = {'display': 'block'}
    elif active_tab == 'charts':
        styles[1] = {'display': 'block'}
    elif active_tab == 'maps':
        styles[2] = {'display': 'block'}
    elif active_tab == 'stats':
        styles[3] = {'display': 'block'}
    elif active_tab == 'insights':
        styles[4] = {'display': 'block'}
    elif active_tab == 'data':
        styles[5] = {'display': 'block'}
    
    return styles

# CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .summary-row {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 20px;
            }
            .summary-card {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                flex: 1;
                min-width: 150px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .report-box {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                margin-bottom: 15px;
            }
            .kpi-row {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 20px;
            }
            .kpi-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .kpi-card:hover {
                transform: translateY(-2px);
            }
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
'''

if __name__ == '__main__':
    app.run(debug=True, port=8050)