
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

# Initialize the Dash app
app = dash.Dash(__name__)

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

def parse_uploaded_file(contents, filename):
    """Parse uploaded CSV file with better encoding handling"""
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

# App layout - ALL components defined upfront
app.layout = html.Div([
    # Data stores
    dcc.Store(id='dataset'),
    dcc.Store(id='analysis-info'),
    
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
            id='csv-upload',
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
        html.Div(id='upload-message', style={'textAlign': 'center', 'marginTop': 10})
    ], style={'marginBottom': 30}),
    
    # Main content area - all components defined here
    html.Div(id='main-area', children=[
        # Navigation tabs
        dcc.Tabs(id='tabs', value='summary', children=[
            dcc.Tab(label='📊 Summary', value='summary'),
            dcc.Tab(label='📈 Charts', value='charts'),
            dcc.Tab(label='🗺️ Maps', value='maps'),
            dcc.Tab(label='📋 Data', value='data')
        ], style={'marginBottom': 20}),
        
        # Tab content area
        html.Div(id='tab-display', children=[
            # Summary tab content
            html.Div(id='summary-tab', children=[], style={'display': 'none'}),
            
            # Charts tab content
            html.Div(id='charts-tab', children=[
                html.H3("📈 Interactive Visualizations", style={'color': '#2c3e50'}),
                
                # Chart controls - all defined here
                html.Div([
                    html.Div([
                        html.Label("Chart Type:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='chart-selector',
                            options=[
                                {'label': 'Histogram', 'value': 'histogram'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Scatter Plot', 'value': 'scatter'},
                                {'label': 'Bar Chart', 'value': 'bar'},
                                {'label': 'Line Chart', 'value': 'line'},
                                {'label': 'Correlation Heatmap', 'value': 'heatmap'}
                            ],
                            value='bar'  # Changed default to bar chart
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                    
                    html.Div([
                        html.Label("X-Axis:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(id='x-axis', value=None)
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                    
                    html.Div([
                        html.Label("Y-Axis:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(id='y-axis', value=None)
                    ], style={'width': '30%', 'display': 'inline-block'})
                ], style={'marginBottom': 10}),
                
                # Chart recommendations
                html.Div(id='chart-recommendations', style={'marginBottom': 10, 'padding': 10, 'backgroundColor': '#e8f4f8', 'borderRadius': 5}),
                
                dcc.Graph(id='chart-output', style={'height': 600})
            ], style={'display': 'none'}),
            
            # Maps tab content
            html.Div(id='maps-tab', children=[
                html.H3("🗺️ Geographic Visualization", style={'color': '#2c3e50'}),
                
                html.P("Create region-based heat maps or coordinate-based scatter maps:", 
                       style={'marginBottom': 20, 'fontStyle': 'italic'}),
                
                # Map type selector
                html.Div([
                    html.Label("Map Type:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='map-selector',
                        options=[
                            {'label': ' Region Heat Map', 'value': 'region'},
                            {'label': ' Coordinate Map', 'value': 'coords'}
                        ],
                        value='region',
                        inline=True,
                        style={'marginBottom': 15}
                    )
                ]),
                
                # Map controls - all defined here
                html.Div([
                    html.Div([
                        html.Label("Location Column:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(id='location-col', value=None)
                    ], style={'width': '22%', 'display': 'inline-block', 'marginRight': '4%'}),
                    
                    html.Div([
                        html.Label("Value Column:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(id='value-col', value=None)
                    ], style={'width': '22%', 'display': 'inline-block', 'marginRight': '4%'}),
                    
                    html.Div([
                        html.Label("Latitude:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(id='lat-col', value=None)
                    ], style={'width': '22%', 'display': 'inline-block', 'marginRight': '4%'}),
                    
                    html.Div([
                        html.Label("Longitude:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(id='lon-col', value=None)
                    ], style={'width': '22%', 'display': 'inline-block'})
                ], style={'marginBottom': 20}),
                
                dcc.Graph(id='map-output', style={'height': 600})
            ], style={'display': 'none'}),
            
            # Data tab content
            html.Div(id='data-tab', children=[
                html.H3("📋 Data Explorer", style={'color': '#2c3e50'}),
                html.Div(id='data-info'),
                html.Div(id='data-display')
            ], style={'display': 'none'})
        ])
    ], style={'display': 'none'})
    
], style={'padding': 20, 'fontFamily': 'Arial, sans-serif'})

# File upload callback
@app.callback(
    [Output('dataset', 'data'),
     Output('analysis-info', 'data'),
     Output('upload-message', 'children'),
     Output('main-area', 'style'),
     Output('x-axis', 'options'),
     Output('y-axis', 'options'),
     Output('location-col', 'options'),
     Output('value-col', 'options'),
     Output('lat-col', 'options'),
     Output('lon-col', 'options'),
     Output('x-axis', 'value'),
     Output('y-axis', 'value'),
     Output('value-col', 'value')],
    [Input('csv-upload', 'contents')],
    [State('csv-upload', 'filename')]
)
def handle_upload(contents, filename):
    if contents is None:
        empty_options = []
        return {}, {}, "", {'display': 'none'}, empty_options, empty_options, empty_options, empty_options, empty_options, empty_options, None, None, None
    
    # Parse file
    df, status_msg = parse_uploaded_file(contents, filename)
    
    if df is None:
        empty_options = []
        return {}, {}, html.Div(status_msg, style={'color': 'red'}), {'display': 'none'}, empty_options, empty_options, empty_options, empty_options, empty_options, empty_options, None, None, None
    
    # Analyze data
    analyzer = DataAnalyzer(df)
    cleaning_report = analyzer.clean_data()
    insights = analyzer.generate_insights()
    
    # Prepare data
    analysis_data = {
        'cleaning_report': cleaning_report,
        'insights': insights,
        'numeric_cols': list(analyzer.df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': list(analyzer.df.select_dtypes(include=['object']).columns)
    }
    
    # Create dropdown options
    all_cols = [{'label': col, 'value': col} for col in analyzer.df.columns]
    numeric_cols = [{'label': col, 'value': col} for col in analysis_data['numeric_cols']]
    
    # Ensure we have numeric columns, if not, use all columns
    if not numeric_cols:
        numeric_cols = all_cols
    
    # Set default values
    default_x = analyzer.df.columns[0] if len(analyzer.df.columns) > 0 else None
    default_y = analyzer.df.columns[1] if len(analyzer.df.columns) > 1 else None
    default_value = analysis_data['numeric_cols'][0] if analysis_data['numeric_cols'] else None
    
    success_msg = html.Div("✅ " + status_msg, style={'color': 'green', 'fontWeight': 'bold'})
    
    return (analyzer.df.to_dict('records'), analysis_data, success_msg, {'display': 'block'}, 
            all_cols, all_cols, all_cols, numeric_cols, numeric_cols, numeric_cols,
            default_x, default_y, default_value)

# Tab display callback
@app.callback(
    [Output('summary-tab', 'style'),
     Output('charts-tab', 'style'),
     Output('maps-tab', 'style'),
     Output('data-tab', 'style'),
     Output('summary-tab', 'children')],
    [Input('tabs', 'value'),
     Input('dataset', 'data'),
     Input('analysis-info', 'data')]
)
def display_tab(active_tab, data, analysis):
    # Set all tabs to hidden first
    hidden = {'display': 'none'}
    visible = {'display': 'block'}
    
    # Determine which tab to show
    summary_style = visible if active_tab == 'summary' else hidden
    charts_style = visible if active_tab == 'charts' else hidden
    maps_style = visible if active_tab == 'maps' else hidden
    data_style = visible if active_tab == 'data' else hidden
    
    # Generate summary content
    summary_content = []
    if data and analysis:
        df = pd.DataFrame(data)
        
        # Summary cards
        summary_content = [
            html.H3("📊 Dataset Summary", style={'color': '#2c3e50'}),
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
                    html.H2(f"{len(analysis['numeric_cols'])}", style={'color': '#27ae60', 'margin': 0, 'textAlign': 'center'}),
                    html.P("Numeric Columns", style={'margin': 0, 'textAlign': 'center'})
                ], className='summary-card')
            ], className='summary-row'),
            
            # Cleaning report
            html.H3("🧹 Data Cleaning Report", style={'color': '#2c3e50', 'marginTop': 30}),
            html.Div([
                html.Ul([html.Li(item) for item in analysis['cleaning_report']])
            ], className='report-box'),
            
            # Insights
            html.H3("💡 Automatic Insights", style={'color': '#2c3e50', 'marginTop': 30}),
            html.Div([
                html.Ul([html.Li(insight) for insight in analysis['insights']])
            ], className='report-box'),
            
            # Column info
            html.H3("📋 Column Information", style={'color': '#2c3e50', 'marginTop': 30}),
            dash_table.DataTable(
                data=[{
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Unique': df[col].nunique(),
                    'Missing': df[col].isnull().sum(),
                    'Missing %': f"{(df[col].isnull().sum() / len(df) * 100):.1f}%"
                } for col in df.columns],
                columns=[
                    {'name': 'Column Name', 'id': 'Column'},
                    {'name': 'Data Type', 'id': 'Type'},
                    {'name': 'Unique Values', 'id': 'Unique'},
                    {'name': 'Missing Values', 'id': 'Missing'},
                    {'name': 'Missing %', 'id': 'Missing %'}
                ],
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                page_size=15
            )
        ]
    
    return summary_style, charts_style, maps_style, data_style, summary_content

# Chart callback
@app.callback(
    Output('chart-output', 'figure'),
    [Input('chart-selector', 'value'),
     Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('dataset', 'data')]
)
def update_chart(chart_type, x_col, y_col, data):
    if not data or not x_col:
        return go.Figure().add_annotation(
            text="Select data and X-axis column to generate chart", 
            showarrow=False, xref="paper", yref="paper", 
            x=0.5, y=0.5, font_size=16
        )
    
    df = pd.DataFrame(data)
    
    try:
        if chart_type == 'histogram':
            fig = px.histogram(df, x=x_col, title=f'Distribution of {x_col}')
            
        elif chart_type == 'box':
            if y_col and y_col != x_col:
                fig = px.box(df, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
            else:
                fig = px.box(df, y=x_col, title=f'Box Plot of {x_col}')
                
        elif chart_type == 'scatter' and y_col and y_col != x_col:
            fig = px.scatter(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
            
        elif chart_type == 'bar':
            if df[x_col].dtype == 'object' and y_col and y_col != x_col:
                # Aggregate data by category and sum the values
                bar_data = df.groupby(x_col)[y_col].sum().reset_index()
                bar_data = bar_data.sort_values(y_col, ascending=False).head(20)
                fig = px.bar(bar_data, x=x_col, y=y_col, 
                           title=f'Total {y_col} by {x_col}')
                fig.update_xaxes(tickangle=45)
            elif df[x_col].dtype == 'object':
                value_counts = df[x_col].value_counts().head(20)
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f'Count of {x_col}',
                           labels={'x': x_col, 'y': 'Count'})
                fig.update_xaxes(tickangle=45)
            else:
                fig = px.histogram(df, x=x_col, title=f'Distribution of {x_col}')
                
        elif chart_type == 'line' and y_col and y_col != x_col:
            # For line charts with categorical X-axis, aggregate and sort data
            if df[x_col].dtype == 'object':
                line_data = df.groupby(x_col)[y_col].sum().reset_index()
                line_data = line_data.sort_values(y_col)  # Sort by Y values for meaningful line
                fig = px.line(line_data, x=x_col, y=y_col, 
                            title=f'Total {y_col} by {x_col} (Sorted by Value)')
                fig.update_xaxes(tickangle=45)
            else:
                fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, 
                            title=f'{y_col} over {x_col}')
                            
        elif chart_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Correlation Heatmap", color_continuous_scale='RdBu')
            else:
                fig = go.Figure().add_annotation(
                    text="Need at least 2 numeric columns for correlation heatmap", 
                    showarrow=False, xref="paper", yref="paper", 
                    x=0.5, y=0.5, font_size=14
                )
        else:
            fig = px.histogram(df, x=x_col, title=f'Distribution of {x_col}')
        
        return fig
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error generating chart: {str(e)}", 
            showarrow=False, xref="paper", yref="paper", 
            x=0.5, y=0.5, font_size=14
        )

# Map callback
@app.callback(
    Output('map-output', 'figure'),
    [Input('map-selector', 'value'),
     Input('location-col', 'value'),
     Input('value-col', 'value'),
     Input('lat-col', 'value'),
     Input('lon-col', 'value'),
     Input('dataset', 'data')]
)
def update_map(map_type, location_col, value_col, lat_col, lon_col, data):
    if not data:
        return go.Figure().add_annotation(
            text="No data available for mapping", 
            showarrow=False, xref="paper", yref="paper", 
            x=0.5, y=0.5, font_size=16
        )
    
    df = pd.DataFrame(data)
    
    try:
        if map_type == 'region' and location_col and value_col:
            # Region-based heat map
            map_data = df.groupby(location_col)[value_col].sum().reset_index()
            map_data = map_data.sort_values(value_col, ascending=False)
            
            # Try different location modes
            location_modes = ['country names', 'ISO-3', 'USA-states']
            fig = None
            
            for mode in location_modes:
                try:
                    fig = go.Figure(data=go.Choropleth(
                        locations=map_data[location_col],
                        z=map_data[value_col],
                        locationmode=mode,
                        colorscale='Blues',
                        text=map_data[location_col],
                        colorbar_title=value_col,
                        hovertemplate=f'<b>%{{text}}</b><br>{value_col}: %{{z:,.0f}}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f'{value_col} by {location_col}',
                        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular')
                    )
                    break  # Success, exit loop
                except Exception:
                    continue  # Try next mode
            
            if fig is None:
                # If choropleth fails, create a bar chart as fallback
                fig = px.bar(map_data.head(20), x=location_col, y=value_col,
                           title=f'Top 20 {location_col} by {value_col} (Geographic map unavailable)')
                fig.update_xaxes(tickangle=45)
            
        elif map_type == 'coords' and lat_col and lon_col:
            # Coordinate-based map
            hover_data = [location_col] if location_col else []
            
            fig = px.scatter_mapbox(
                df,
                lat=lat_col,
                lon=lon_col,
                color=value_col if value_col else None,
                size=value_col if value_col else None,
                hover_data=hover_data,
                title="Geographic Distribution",
                mapbox_style="open-street-map",
                zoom=3,
                height=600
            )
        else:
            instruction = "⚙️ Select appropriate columns:\n\n"
            if map_type == 'region':
                instruction += "✓ Location Column: Choose column with region/country names\n"
                instruction += "✓ Value Column: Choose numeric column for heat map intensity\n\n"
                instruction += "💡 Tip: Works best with country names, state names, or ISO codes"
            else:
                instruction += "✓ Latitude Column: Choose column with latitude coordinates\n"
                instruction += "✓ Longitude Column: Choose column with longitude coordinates\n"
                instruction += "• Optional: Value Column for color/size coding"
            
            fig = go.Figure()
            fig.add_annotation(
                text=instruction, 
                showarrow=False, xref="paper", yref="paper", 
                x=0.5, y=0.5, font_size=14, align="left"
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )
        
        return fig
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error generating map: {str(e)}\n\nTip: Ensure location names match standard country/state names", 
            showarrow=False, xref="paper", yref="paper", 
            x=0.5, y=0.5, font_size=14
        )

# Chart recommendations callback
@app.callback(
    Output('chart-recommendations', 'children'),
    [Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('dataset', 'data')]
)
def update_chart_recommendations(x_col, y_col, data):
    if not data or not x_col:
        return ""
    
    df = pd.DataFrame(data)
    recommendations = []
    
    x_is_categorical = df[x_col].dtype == 'object'
    y_is_categorical = y_col and df[y_col].dtype == 'object' if y_col else False
    y_is_numeric = y_col and df[y_col].dtype in ['int64', 'float64'] if y_col else False
    
    if x_is_categorical and y_is_numeric:
        recommendations.append("💡 Recommended: Bar Chart - Perfect for showing numeric values by category")
    elif not x_is_categorical and y_is_numeric:
        recommendations.append("💡 Recommended: Scatter Plot or Line Chart - Great for numeric relationships")
    elif x_is_categorical and not y_col:
        recommendations.append("💡 Recommended: Bar Chart - Shows frequency of categories")
    elif not x_is_categorical and not y_col:
        recommendations.append("💡 Recommended: Histogram - Shows distribution of numeric data")
    
    if x_is_categorical and y_is_numeric:
        recommendations.append("⚠️ Line Chart may not be meaningful with categorical X-axis")
    
    if recommendations:
        return html.Div([
            html.P("📊 Chart Suggestions:", style={'fontWeight': 'bold', 'margin': 0, 'color': '#2c3e50'}),
            html.Ul([html.Li(rec, style={'fontSize': 12}) for rec in recommendations], style={'margin': 5})
        ])
    
    return ""

# Data display callback
@app.callback(
    [Output('data-info', 'children'),
     Output('data-display', 'children')],
    [Input('dataset', 'data')]
)
def update_data_display(data):
    if not data:
        return "", ""
    
    df = pd.DataFrame(data)
    
    info = html.P(f"Showing first 100 rows of {len(df)} total rows", 
                  style={'fontStyle': 'italic', 'marginBottom': 15})
    
    table = dash_table.DataTable(
        data=df.head(100).to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        style_cell={'textAlign': 'left', 'padding': '8px', 'maxWidth': '150px', 'overflow': 'hidden'},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        page_size=20,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'}
    )
    
    return info, table

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