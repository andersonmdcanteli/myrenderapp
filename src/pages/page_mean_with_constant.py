### ------ IMPORTS ------ ###

import base64
import datetime
import io

# --- dash --- #
from dash import callback, dash_table, dcc, html, Input, Output, State
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from dash.dash import no_update

# --- Third part --- #
import pandas as pd
import numpy as np
from scipy.stats import t as t_student
from scipy.stats import shapiro as shapiro_wilk
from scipy.stats import anderson as anderson_norm
from pandas.api.types import is_numeric_dtype

### ------ datasets ------ ####
from functions import outliers as func_outliers

### ------ database language ------ ###









### ------ Configs ------ ###
configuracoes_grafico = {
    'staticPlot': False,     # True, False
    'scrollZoom': True,      # True, False
    'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
    'showTips': True,       # True, False
    'displayModeBar': True,  # True, False, 'hover'
    'watermark': True,
    'modeBarButtonsToRemove': ['lasso2d'],
}


### ------ data for dropdown ------ ###
alpha_options = [
    {'label': '1%', 'value': 0.01},
    {'label': '5%', 'value': 0.05},
    {'label': '10%', 'value': 0.1},
]

normalidade_options = [
    {'label': 'Shapiro-Wilk', 'value': 'Shapiro-Wilk'},
    {'label': 'Anderson-Darling', 'value': "Anderson-Darling"},
]

outliers_options = [
    {'label': 'Grubbs', 'value': 'Grubbs'},
    {'label': 'Dixon', 'value': "Dixon"},
]

decimal_options = []
for i in range(10):
    decimal_options.append({'label': str(i), 'value': i})


test_options = [
    {'label': 'Unilateral', 'value': 1},
    {'label': 'Bilateral', 'value': 2},
]


### ------ styling dash table ------ ###

style_data = {
    'whiteSpace': 'normal',
    'height': 'auto'
    }

style_cell = {
    'font_size': 'clamp(1rem, 0.5vw, 0.5vh)',
    'textAlign': 'center',
    'height': 'auto',
    'minWidth': '80px',
    'width': '80px',
    'maxWidth': '80px',
    'whiteSpace': 'normal'
}

style_table = {
    'overflowX': 'auto',
    'overflowY': 'auto',
    'maxHeight': '500px',
    }

style_header = {
    'fontWeight': 'bold', # deixando em negrito
    }



### ------- datasets ------ ###
df_generic_data = pd.DataFrame({
    "Sample": (1, 2, 3, 4, 5),
    "Data": (0.01, 0.0, 0.03, -0.01, 0.04)
})

### ------ FUNCTIONS ------ ###

def make_column_type(df):
    columns_options = []
    for col in df.columns:
        col_options = {"name": col, "id": col}
        if df[col].dtype != object:
            col_options["type"] = "numeric" # para permitir filtrar numéricamente
            # col_options['format'] = Format(precision=4, scheme=Scheme.fixed)
        columns_options.append(col_options)
    return columns_options


def parse_contents(contents, filename):#, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if str(filename).endswith(".csv"):
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            return df
        # elif 'xls' in filename:
        #     # Assume that the user uploaded an excel file
        #     df = pd.read_excel(io.BytesIO(decoded))
    except:
        return 'There was an error processing this file.'
    return 'There was an error processing this file.'









### ------ LAYOUTS ------ ###

# --- MENU --- #
offcanvas = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-cog fa-2x contact_icons"),
            id="mean-with-constant-open-offcanvas", n_clicks=0, className="btn bg-transparent rounded-circle border-white shadow-none",
            ),
        dbc.Offcanvas(
            children = [
                    dbc.Row([
                        dbc.Col(
                            html.Label(id="mean-with-constant-text-significance-level"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dcc.Dropdown(
                                        id='mean-with-constant-alpha-picker',
                                        value = alpha_options[1]['value'],
                                        options = alpha_options,
                                        clearable = False,
                                        persistence=True,
                                        persistence_type="local",
                                        ),
                                    )
                                ),
                            ]),
                        ],
                    ),
                    dbc.Row([
                        dbc.Col(
                            html.Label(id="mean-with-constant-text-norm-test"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dcc.Dropdown(
                                        id='mean-with-constant-normality-picker',
                                        value = normalidade_options[0]['value'],
                                        options = normalidade_options,
                                        clearable = False,
                                        persistence=True,
                                        persistence_type="local",
                                        ),
                                    )
                                ),
                            ]),
                        ],
                    ),
                    dbc.Row([
                        dbc.Col(
                            html.Label(id="mean-with-constant-text-outlier-test"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dcc.Dropdown(
                                        id='mean-with-constant-outlier-picker',
                                        value = outliers_options[0]['value'],
                                        options = outliers_options,
                                        clearable = False,
                                        persistence=True,
                                        persistence_type="local",
                                        ),
                                    )
                                ),
                            ]),
                        ],
                    ),
                    dbc.Row([
                        dbc.Col(
                            html.Label(id="mean-with-constant-text-decimal-places"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dcc.Dropdown(
                                        id='mean-with-constant-decimal-places-picker',
                                        value = decimal_options[3]['value'],
                                        options = decimal_options,
                                        clearable = False,
                                        persistence=True,
                                        persistence_type="local",
                                        ),
                                    )
                                ),
                            ]),
                        ],
                    ),
                    html.Hr(),
                    dbc.Row(
                        dbc.Col(
                            id="mean-with-constant-offcanvas-test-summary"
                        )
                    ),
                ],
            id="mean-with-constant-offcanvas",
            # title="preferences",
            is_open=False,
        ),
    ]
)










### ------ LAYOUTS ------ ###
# --- MAIN LAYOUT --- #
layout = html.Div([
    # General title
    dbc.Row([
        dbc.Col(
            sm = 0,
            lg = 1,
            align="center"
        ),
        dbc.Col(
            html.H2(id='mean-with-constant-title'),
            width = {"size": 10},
            align = "center"
        ),
        dbc.Col(
            offcanvas,
            width = {"size": 1},
            align="center"
        ),
    ], style = {'textAlign': 'center', 'paddingTop': '30px', 'paddingBottom': '30px'},),
    # Button to load the data
    dbc.Row(
        dbc.Col(
            dcc.Upload(
                id='mean-with-constant-data-upload',
                children=html.Div([
                    dbc.Button(id="mean-with-constant-text-drag-drop", outline=True, color="success", className="me-1", size="lg"),
                ]),
                style={
                    'textAlign': 'center',
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
        ), style={"paddingBottom": "25px"}
    ),
    # Hidden div to show # WARNINGs
    dbc.Row(
        dbc.Col(
            html.Div(id='mean-with-constant-alert'),
        ), style={"textAlign": "center", "margin": "10px"}
    ),
    # Img Div to show the file tips
    dbc.Row(
        dbc.Col(
            id="mean-with-constant-image-descritption"
        ), style={"paddingTop": "20px", "paddingBottom": "20px", "textAlign": "center"}
    ),
    dbc.Row(
        dbc.Col(
            html.Div(
                [
                    dbc.Button(id="mean-with-constant-download-button", outline=True, color="success", className="me-1", size="sm"),
                    dcc.Download(id="mean-with-constant-download-csv"),
                ], style={"textAlign": "center"}, id="mean-with-constant-download-div"),
        ), justify="center", style={"textAlign": "center", "paddingBottom": "25px", "paddingTop": "15px"},
    ),
    # Main content
    html.Div(
        [
        dbc.Row([
            # Column to enter and show data
            dbc.Col([
                # Table to show the data supplied
                dbc.Row(
                    dbc.Col([
                        dbc.Row(
                            dbc.Col(
                                html.H5(id="mean-with-constant-data-name")
                            ),
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.Div(id='mean-with-constant-datatable'),
                            ),
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.H4(id="mean-with-constant-text-data-summary"), style={"textAlign": "center", "paddingTop": "25px"}
                            )
                        ),
                        # summary table
                        dbc.Row(
                            dbc.Col(
                                id="mean-with-constant-datasummary"
                            ), style={"paddingBottom": "15px"}
                        ),
                    ])
                )
            ], lg=2),
            # Column to show results
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Row(
                            dbc.Col([
                                # test type
                                dbc.Row([
                                    dbc.Col(
                                        html.H5(
                                        children=[
                                            html.Span(id="mean-with-constant-text-test-type"),
                                            html.Span(":"),

                                        ])
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id='mean-with-constant-test-type',
                                            value = test_options[1]['value'],
                                            options = test_options,
                                            clearable = False,
                                            persistence=True,
                                            persistence_type="session",
                                            ),
                                    )
                                ]),
                                # Know value
                                dbc.Row([
                                    dbc.Col(
                                        html.H5([
                                            html.Span(id='mean-with-constant-text-known-value'),
                                            html.Span(":"),
                                        ])
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id="mean-with-constant-know-value", type="number", debounce=True, placeholder=0,
                                            persistence=True,
                                            persistence_type="session",
                                            )
                                    )
                                ]),
                            ],
                            ), style={"paddingTop": "5px", "paddingBottom": "25px"}
                        ),
                        # test Resuts
                        dbc.Row([
                            dbc.Col([
                                dbc.Row(
                                    dbc.Col(
                                        html.H3(id="mean-with-constant-text-test-result")
                                    ), style={"textAlign": "center"}
                                ),
                                dbc.Row(
                                    dbc.Col(
                                        id='mean-with-constant-table-results'
                                    )
                                ),
                            ]),
                        ]),
                    ], lg=4),
                    dbc.Col([
                        dbc.Row(
                            dbc.Col(
                                html.H4(id="mean-with-constant-text-student-plot"), style={"textAlign": "center"}
                            )
                        ),
                        # Student's t plot
                        dbc.Row(
                            dbc.Col(
                                dcc.Graph(id='mean-with-constant-student-plot', mathjax=True, config=configuracoes_grafico),
                            ), justify="center"
                        ),
                    ], lg=8)
                ]),
                dbc.Row(
                    dbc.Col([
                        dbc.Row(
                            html.H3(id="mean-with-constant-text-result"), style={"textAlign": "center"}
                        ),
                        dbc.Row(
                            html.H4(id="mean-with-constant-result"), style={"textAlign": "center", 'margin': "10px"}
                        ),
                    ], style={"paddingTop": "25px", "paddingBottom": "25px"})
                )
            ], lg=10),
        ]),
        ], style= {'display': 'block'}, id="mean-with-constant-main-div"
    ),

    dcc.Store(id='mean-with-constant-data-store', storage_type="local"),


],)





### ------ CALLBACKS ------ ###

### ------ ------ ###
@callback(
        Output('mean-with-constant-data-name', 'children'),
        Output('mean-with-constant-alert', 'children'),
        Output('mean-with-constant-data-store', 'data'),
        Output('mean-with-constant-image-descritption', 'children'),
        Output('mean-with-constant-main-div', component_property='style'),
        Output('mean-with-constant-download-div', component_property='style'),

        Input('mean-with-constant-data-upload', 'contents'),
        State('mean-with-constant-data-upload', 'filename'),
        Input('language-picker', 'value'),
)
def update_output(list_of_contents, list_of_names, lang):
    block = 'none'
    if list_of_names is None:
        if lang == "en":
            src = "assets/mean-with-constant/descritption-en.png"
            alt = "description of how the csv file should be formatted"
        else:
            src = "assets/mean-with-constant/descritption-pt-br.png"
            alt = "descrição de como o arquivo csv deve ser formatado"

        image = html.Img(src=src, alt=alt, style={"width": "100%", "maxWidth": "1200px", "height": "auto"})
        df = pd.DataFrame({})
        return "Generic Data", [""] , df.to_json(date_format='iso', orient='split'), image, {'display': block}, {'display': 'block'}
    else:
        if len(list_of_names) != 1:
            if lang == "en":
                textos = ("Invalid value!", "We received", "files, but only a single file with the data is accepted.", "Please, send a single file.", "Files received:")
            else:
                textos = ("Valor inválido!", "Recebemos", "arquivos, mas só é aceito um único arquivo com os dados.", "Por favor, envie um único arquivo.", "Arquivos recebidos:")

            nomes = []
            for nome in list_of_names:
                nomes.append(html.Ul(
                                html.Li(nome),
                            ))

            alert = dbc.Alert([
                html.H4(textos[0], className="alert-heading"),
                html.P(f"{textos[1]} {len(list_of_names)} {textos[2]}"),
                html.H5(textos[3]),
                html.P(textos[4]),
                html.Span(nomes)
            ], color="danger", dismissable=True)
            return no_update, alert , no_update, no_update, {'display': block}, {'display': 'block'}
        else:
            if len(list_of_contents[0]) < 6:
                if lang == "en":
                    textos = ("There was an error processing this file!", "The", "file contains no data!")
                else:
                    textos = ("Ocorreu um erro ao processar este arquivo!", "O arquivo", "não contém dados!")

                alert = dbc.Alert([
                    html.H4(textos[0], className="alert-heading"),
                    html.P(f"{textos[1]} '{list_of_names[0]}' {textos[2]}"),
                ], color="danger", dismissable=True)
                return no_update, alert , no_update, no_update, {'display': block}, {'display': 'block'}


            df = parse_contents(list_of_contents[0], list_of_names[0])
            if type(df) == str:
                if lang == "en":
                    textos = ("There was an error processing this file!", "Unable to get data from file", "Hint:", "Only '.csv' files are accepted")
                else:
                    textos = ("Ocorreu um erro ao processar este arquivo!", "Não foi possível obter os dados do arquivo", "Dica:", "É aceito apenas arquivos '.csv'")

                alert = dbc.Alert([
                    html.H4(textos[0], className="alert-heading"),
                    html.P(f"{textos[1]} '{list_of_names[0]}'"),
                    html.H5(textos[2]),
                    html.P(textos[3]),

                ], color="danger", dismissable=True)

                return no_update, alert , no_update, no_update, {'display': block}, {'display': 'block'}
            else:
                if df.shape[1] < 2:
                    if lang == "en":
                        textos = ("There was an error processing this file!", "The number of columns is incompatible", "The file must contain 2 columns:", "The first must contain an identification for each sample;", "The second must contain the respective observations;")
                    else:
                        textos = ("Ocorreu um erro ao processar este arquivo!", "O número de colunas é incompatível", "O arquivo deve conter 2 colunas:", "A primeira deve conter uma identifcação para cada amostra;", "A segunda deve conter as observações;")

                    alert = dbc.Alert([
                        html.H4(textos[0], className="alert-heading"),
                        html.P(textos[1]),
                        html.H5(textos[2]),
                        html.Ul([
                            html.Li(textos[3]),
                            html.Li(textos[4]),
                        ])
                    ], color="danger", dismissable=True)
                    return no_update, alert , no_update, no_update, {'display': block}, {'display': 'block'}

                elif df.shape[1] > 2:
                    if lang == "en":
                        textos = ("Lots of columns!", "The given file ", "contains", "columns, but only 2 are needed.", "The test will be performed using the second column")
                    else:
                        textos = ("Muitas colunas!", "O arquivo fornecido ", "contém", "colunas, mas apenas 2 são necessárias.", "O teste será realizado utilizando a segunda coluna")

                    alert = dbc.Alert([
                        html.H4(textos[0], className="alert-heading"),
                        html.P(f"{textos[1]} ('{list_of_names[0]}') {textos[2]} {str(df.shape[1])} {textos[3]}"),
                        html.P(f"{textos[4]} ('{df.columns[1]}') "),
                    ], color="warning", dismissable=True)
                    data_name = list_of_names[0]

                else:
                    alert = [""]


    if not is_numeric_dtype(df[df.columns[1]]):
        if lang == "en":
            textos = ("Non-numerical value!", "At least one entry in column", "is not numeric!", "Hint:", "The decimal point separator is the dot ('.')")
        else:
            textos = ("Valor não numérico!", "Pelo menos uma entrada da coluna", "não é numérica!", "Dica:", "O separador de casas decimais é o ponto ('.')")

        alert = dbc.Alert([
            html.H4(textos[0], className="alert-heading"),
            html.P(f"{textos[1]} '{df.columns[1]}' {textos[2]}"),
            html.H5(textos[3]),
            html.P(textos[4]),
        ], color="danger", dismissable=True)
        return no_update, alert, no_update, no_update, {'display': block}, {'display': 'block'}


    if df[df.columns[1]].isnull().values.any():
        if lang == "en":
            textos = ("Missing value!", "At least one entry in column", "is empty!")
        else:
            textos = ("Valor faltante!", "Pelo menos uma entrada da coluna", "esta vazia!")

        alert = dbc.Alert([
            html.H4(textos[0], className="alert-heading"),
            html.P(f"{textos[1]} '{df.columns[1]}' {textos[2]}"),
        ], color="danger", dismissable=True)
        return no_update, alert, no_update, no_update, {'display': block}, {'display': 'block'}

    if df.shape[0] < 3:
        if lang == "en":
            textos = ("Small sample size!", "A minimum of 3 observations are needed to apply the test, but the dataset contains only")
        else:
            textos = ("Tamanho amostral pequeno!", "São necessários no mínimo 3 observações para aplicar o teste, mas o conjunto de dados contém apenas")

        alert = dbc.Alert([
            html.H4(textos[0], className="alert-heading"),
            html.P(f"{textos[1]} '{df.shape[0]}'"),
        ], color="danger", dismissable=True)
        return no_update, alert, no_update, no_update, {'display': block}, {'display': 'block'}

    return list_of_names[0], alert, df.to_json(date_format='iso', orient='split'), "", {'display': 'block'}, {'display': 'none'}



### ------ ------ ###
@callback(Output('mean-with-constant-datatable', 'children'),
              Input('mean-with-constant-data-store', 'data'),
              )
def update_data_table(df_json):
    df = pd.read_json(df_json, orient='split').copy()
    table = dash_table.DataTable(
                    columns = make_column_type(df),
                    data = df.to_dict('records'),
                    style_data = style_data,
                    style_table = style_table,
                    style_cell = style_cell,
                    sort_action="native",
                    style_header = style_header
                    ),



    return table


### ------ ------ ###
@callback(
        Output('mean-with-constant-datasummary', 'children'),
        Input('mean-with-constant-data-store', 'data'),
        Input('mean-with-constant-alpha-picker', 'value'),
        Input("mean-with-constant-normality-picker", 'value'),
        Input("mean-with-constant-outlier-picker", 'value'),
        Input('mean-with-constant-decimal-places-picker', 'value'),
        Input('language-picker', 'value'),
              )
def update_data_summary(df_json, alfa, normality, outlier_test, dec_places, lang):

    if lang == "en":
        textos = ("Metric", "Mean", "Standart deviation", "Variance", "Confidence interval", "CV (%)", "Normality", "Outliers",
                "Normal", "Not Normal", 'p-value', "Confidence interval, at", "Coefficient of variation", "Statistic", "Critical",
                "Yes", "No", "Division by zero error")

    else:
        textos = ("Métricas", "Média", "Desvio padrão", "Variância", "Intervalo de Confiança", "CV (%)", "Normalidade", "Outliers",
        "Normal", "Não Normal", "p-valor", "Intervalo de confiança, com", "Coeficiente de variação", "Estatística", "Crítico",
        "Sim", "Não", "Erro de divisão por zero")


    df = pd.read_json(df_json, orient='split').copy()
    if df.empty:
        return [""]

    column = df.columns[1]
    n_size = df.shape[0]
    mean = np.mean(df[column])
    std = np.std(df[column], ddof=1)
    variance = np.var(df[column], ddof=1)
    conf_interval = t_student.ppf(1-alfa/2, n_size - 1)*std/np.sqrt(n_size)
    if mean == 0:
        cv = None
        cv_texto = textos[17]
    else:
        cv = 100*std/mean
        cv_texto = str(cv)

    # normality test
    if normality == "Shapiro-Wilk":
        norm = shapiro_wilk(df[column])
        if norm.pvalue < alfa:
            norm_result = textos[9]
        else:
            norm_result = textos[8]
        norm_texto = f"{textos[10]}={norm.pvalue}"

    else:
        norm = anderson_norm(df[column], 'norm')
        if alfa == 0.10:
            ad_critico = norm[1][1]
        elif alfa == 0.05:
            ad_critico = norm[1][2]
        else:
            ad_critico = norm[1][4]
        if norm[0] > ad_critico:
            norm_result = textos[9]
            norm_texto = f"{textos[13]} ({norm[0]}) > {textos[14]} ({ad_critico})"
        else:
            norm_result = textos[8]
            norm_texto = f"{textos[13]} ({norm[0]}) < {textos[14]} ({ad_critico})"



    if outlier_test == "Grubbs":
        outlier_statistic, outlier_critical, possible_outlier, erro = func_outliers.grubbs(df[column], alfa)
        if not erro:
            if outlier_statistic <= outlier_critical:
                out_result = textos[16]
                outlier_texto = f"{textos[13]} ({outlier_statistic}) < {textos[14]} ({outlier_critical})"
            else:
                out_result = textos[15]
                outlier_texto = f"{textos[13]} ({outlier_statistic}) > {textos[14]} ({outlier_critical})"
        else:
            out_result = "None"
            outlier_texto = textos[17]

    else:
        outlier_statistic, outlier_critical, possible_outlier, erro = func_outliers.dixon(df[column], alfa)
        if not erro:
            if outlier_statistic <= outlier_critical:
                out_result = textos[16]
                outlier_texto = f"{textos[13]} ({outlier_statistic}) < {textos[14]} ({outlier_critical})"
            else:
                out_result = textos[15]
                outlier_texto = f"{textos[13]} ({outlier_statistic}) > {textos[14]} ({outlier_critical})"
        else:
            out_result = "None"
            outlier_texto = textos[17]



    df_summary = pd.DataFrame({
        textos[0]: (textos[1], textos[2], textos[3], textos[4], textos[5], textos[6], textos[7]),
        df.columns[1]: (
                round(mean, dec_places), round(std, dec_places), round(variance, dec_places), round(conf_interval, dec_places), round(cv, dec_places),
                norm_result, out_result
                ),
    })


    tooltip_data = [
        {textos[0]: {'value': textos[1], 'type': 'markdown'}, df.columns[1]: {'value': str(mean), 'type': 'markdown'}},
        {textos[0]: {'value': textos[2], 'type': 'markdown'}, df.columns[1]: {'value': str(std), 'type': 'markdown'}},
        {textos[0]: {'value': textos[3], 'type': 'markdown'}, df.columns[1]: {'value': str(variance), 'type': 'markdown'}},
        {textos[0]: {'value': f'{textos[11]} α = {100*alfa}%', 'type': 'markdown'}, df.columns[1]: {'value': str(conf_interval), 'type': 'markdown'}},
        {textos[0]: {'value': textos[12], 'type': 'markdown'}, df.columns[1]: {'value': cv_texto, 'type': 'markdown'}},
        {textos[0]: {'value': normality, 'type': 'markdown'}, df.columns[1]: {'value': norm_texto, 'type': 'markdown'}},
        {textos[0]: {'value': outlier_test, 'type': 'markdown'}, df.columns[1]: {'value': outlier_texto, 'type': 'markdown'}}]

    table = dash_table.DataTable(
                    columns = make_column_type(df_summary),
                    data = df_summary.to_dict('records'),
                    style_data = style_data,
                    style_table = style_table,
                    style_cell = style_cell,
                    style_header = style_header,
                    tooltip_delay=0,
                    tooltip_duration=None,
                    tooltip_data=tooltip_data
                    ),




    return table



### ------ ------ ###
@callback(
        Output('mean-with-constant-student-plot', 'figure'),
        Output('mean-with-constant-table-results', 'children'),
        Output('mean-with-constant-result', 'children'),
        Input('mean-with-constant-data-store', 'data'),
        Input('mean-with-constant-alpha-picker', 'value'),
        Input('mean-with-constant-decimal-places-picker', 'value'),
        Input('mean-with-constant-know-value', 'value'),
        Input('mean-with-constant-test-type', 'value'),
        Input('language-picker', 'value'),
              )
def mean_comparison_test(df_json, alfa, dec_places, known_value, test_type, lang):

    if lang == "en":
        textos = ("No", "Yes", "Metric", "t-calculated", "t-critical", "p-value", "Hypothesis test", "Result", "Probability", "Student's t-distribution",
            "As the p-value", "is greater than the adopted significance level", "is smaller than the adopted significance level",
            "we have evidence to reject the null hypothesis, and we can say with", "we have no evidence to reject the null hypothesis, and we can state with",
            "confidence that the mean", "is different from", "is equal to",
        )

    else:
        textos = ("Não", "Sim", "Métrica", "t-calculado", "t-crítico", "p-valor", "Teste de hipótese", "Resultado", "Probabilidade", "Distribuição t de Student",
        "Como o p-valor", "é maior do que o nível de significância adotado", "é menor do que o nível de significância adotado",
        "temos evidências para rejeitar a hipótese nula, e podemos afirmar com", "não temos evidências para rejeitar a hipótese nula, e podemos afirmar com",
        "de confiança que a média", "é diferente de", "é igual a"
        )



    if known_value is None:
        known_value = 0

    df = pd.read_json(df_json, orient='split').copy()
    if df.empty:
        return {}, "", ""

    column = df.columns[1]
    n_size = df.shape[0]

    mean = np.mean(df[column])
    std = np.std(df[column], ddof=1)
    variance = np.var(df[column], ddof=1)

    t_calc = (mean - known_value)/(std/np.sqrt(n_size))

    if test_type == 1:
        p_value = 1 - t_student.cdf(np.abs(t_calc), n_size-1)
        if t_calc < 0:
            t_tab = t_student.ppf(alfa, n_size-1)
            hipotese = "H0: X̄ = μ, &nbsp; &nbsp; &nbsp; H1: X̄ < μ"

        else:
            t_tab = t_student.ppf(1-alfa, n_size-1)
            hipotese = "H0: X̄ = μ, &nbsp; &nbsp; &nbsp; H1: X̄ > μ"

    else:
        t_tab = t_student.ppf(1-alfa/2, n_size-1)
        p_value = (1 - t_student.cdf(np.abs(t_calc), n_size-1))*2
        hipotese = "H0: X̄ = μ, &nbsp; &nbsp; &nbsp; H1: X̄ ≠ μ"

    if p_value < alfa:
        x_lim = np.abs(t_calc)*1.5
        result = dcc.Markdown(f"{textos[10]} ({round(p_value, dec_places)}) {textos[12]} ({alfa}), {textos[13]} {100*(1-alfa)}% {textos[15]} ({round(mean, dec_places)}) {textos[16]} {known_value}.",mathjax=True)
        if test_type == 1:
            if t_calc < 0:
                accept = "H1: X̄ < μ"
            else:
                accept = "H1: X̄ > μ"
        else:
            accept = "H1: X̄ ≠ μ"
    else:
        accept = "H0: X̄ = μ"
        x_lim = np.abs(t_tab)*1.5
        result = dcc.Markdown(f"{textos[10]} ({round(p_value, dec_places)}) {textos[11]} ({alfa}), {textos[14]} {100*(1-alfa)}% {textos[15]} ({round(mean, dec_places)}) {textos[17]} {known_value}.",mathjax=True)

    df_result = pd.DataFrame({
        textos[2]: (textos[3], textos[4], textos[5], textos[6]),
        textos[7]: (round(t_calc, dec_places), round(t_tab, dec_places), round(p_value, dec_places), accept),
    })



    tooltip_data=[
        {textos[2]: {'value': textos[3], 'type': 'markdown'}, textos[7]: {'value': str(t_calc), 'type': 'markdown'}},
        {textos[2]: {'value': textos[4], 'type': 'markdown'}, textos[7]: {'value': str(t_tab), 'type': 'markdown'}},
        {textos[2]: {'value': textos[5], 'type': 'markdown'}, textos[7]: {'value': str(p_value), 'type': 'markdown'}},
        {textos[2]: {'value': hipotese, 'type': 'markdown'}, textos[7]: {'value': accept, 'type': 'markdown'}}]

    table = dash_table.DataTable(
                    columns = make_column_type(df_result),
                    data = df_result.to_dict('records'),
                    style_data = style_data,
                    style_table = style_table,
                    style_cell = style_cell,
                    style_header = style_header,
                    tooltip_delay=0,
                    tooltip_duration=None,
                    tooltip_data=tooltip_data
                    ),



    if test_type == 1:
        if t_calc < 0:
            x = np.linspace(-1*x_lim, x_lim, 1000)
            y = t_student.pdf(x, n_size-1, loc=0, scale=1)
            try:
                fig = px.line(x=x, y=y).update_traces(line=dict(color='black'))
            except:
                fig = px.line(x=x, y=y).update_traces(line=dict(color='black'))
            fig['data'][0]['showlegend'] = True
            fig['data'][0]['name'] = textos[9]


            # adding fill probablility
            x = np.linspace(-1*x_lim, t_calc, 1000)
            y = t_student.pdf(x, n_size-1, loc=0, scale=1)
            fig.add_trace(
                go.Scatter(
                    x = x, y=y, fill='tozeroy', mode='none',
                    name=textos[8], legendgroup='1', showlegend=True, fillcolor='salmon',
                    hovertemplate = textos[8] + f" = {np.round(p_value, dec_places)}<extra></extra>"
                )
            )

            # adding critical t left tail
            x = [t_tab, t_tab]
            fig.add_trace(
                go.Scatter(
                    x=x, y= [0, t_student.pdf(0, n_size-1, loc=0, scale=1)],
                    mode='lines', line = dict(color = 'blue', dash="dot", width=1),
                    name=textos[4], legendgroup='2', showlegend=False,
                    customdata= np.round(x, dec_places),
                    hovertemplate = textos[4] + " = %{customdata} <extra></extra>"
                )
            )

            # adding t calc
            x = [t_calc, t_calc]
            fig.add_trace(
                go.Scatter(
                    x=x, y= [0, t_student.pdf(np.abs(t_calc), n_size-1, loc=0, scale=1)],
                    mode='lines', line = dict(color = 'red', dash="dot", width=2),
                    name=textos[3], legendgroup='3', showlegend=True,
                    customdata= np.round(x, dec_places),
                    hovertemplate =  textos[3] + " = %{customdata} <extra></extra>"
                )
            )
            fig.update_layout(
                            legend = dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="right",
                                x=0.99,
                            )
                        )


        else:
            x = np.linspace(-1*x_lim, x_lim, 1000)
            y = t_student.pdf(x, n_size-1, loc=0, scale=1)
            try:
                fig = px.line(x=x, y=y).update_traces(line=dict(color='black'))
            except:
                fig = px.line(x=x, y=y).update_traces(line=dict(color='black'))
            fig['data'][0]['showlegend'] = True
            fig['data'][0]['name'] = textos[9]


            # adding fill probablility
            x = np.linspace(t_calc, x_lim, 1000)
            y = t_student.pdf(x, n_size-1, loc=0, scale=1)
            fig.add_trace(
                go.Scatter(
                    x = x, y=y, fill='tozeroy', mode='none',
                    name=textos[8], legendgroup='1', showlegend=True, fillcolor='salmon',
                    hovertemplate = textos[8] + f" = {np.round(p_value, dec_places)}<extra></extra>"
                )
            )

            # adding critical t right tail
            x = [t_tab, t_tab]
            fig.add_trace(
                go.Scatter(
                    x=x, y= [0, t_student.pdf(0, n_size-1, loc=0, scale=1)],
                    mode='lines', line = dict(color = 'blue', dash="dot", width=1),
                    name=textos[4], legendgroup='2', showlegend=False,
                    customdata= np.round(x, dec_places),
                    hovertemplate = textos[4] + " = %{customdata} <extra></extra>"
                )
            )

            # adding t calc
            x = [t_calc, t_calc]
            fig.add_trace(
                go.Scatter(
                    x=x, y= [0, t_student.pdf(np.abs(t_calc), n_size-1, loc=0, scale=1)],
                    mode='lines', line = dict(color = 'red', dash="dot", width=2),
                    name=textos[3], legendgroup='3', showlegend=True,
                    customdata= np.round(x, dec_places),
                    hovertemplate =  textos[3] + " = %{customdata} <extra></extra>"
                )
            )


            fig.update_layout(
                            legend = dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,

                            )
                        )
    else:
        # adding t distribution with mean = 0 and var = 1
        x = np.linspace(-1*x_lim, x_lim, 1000)
        y = t_student.pdf(x, n_size-1, loc=0, scale=1)

        try:
            fig = px.line(x=x, y=y).update_traces(line=dict(color='black'))
        except:
            fig = px.line(x=x, y=y).update_traces(line=dict(color='black'))
        fig['data'][0]['showlegend'] = True
        fig['data'][0]['name'] = textos[9]

        # adding fill probablility
        x = np.linspace(-1*x_lim, -1*np.abs(t_calc), 1000)
        y = t_student.pdf(x, n_size-1, loc=0, scale=1)
        fig.add_trace(
            go.Scatter(
                x = x, y=y, fill='tozeroy', mode='none',
                name=textos[8], legendgroup='1', showlegend=True, fillcolor='salmon',
                hovertemplate = textos[8] + f" = {np.round(p_value, dec_places)}<extra></extra>"
            )
        )
        # adding fill probablility
        x = np.linspace(np.abs(t_calc), x_lim, 1000)
        y = t_student.pdf(x, n_size-1, loc=0, scale=1)
        fig.add_trace(
            go.Scatter(
                x = x, y=y, fill='tozeroy', mode='none',
                name=textos[8], legendgroup='1', showlegend=False, fillcolor='salmon',
                hovertemplate = textos[8] + f" = {np.round(p_value, dec_places)}<extra></extra>"
            )
        )

        # adding critical t right tail
        x = [t_tab, t_tab]
        fig.add_trace(
            go.Scatter(
                x=x, y= [0, t_student.pdf(0, n_size-1, loc=0, scale=1)],
                mode='lines', line = dict(color = 'blue', dash="dot", width=1),
                name=textos[4], legendgroup='2',
                customdata= np.round(x, dec_places),
                hovertemplate = textos[4] + " = %{customdata} <extra></extra>"
            )
        )
        # adding critical t left tail
        x = [-1*t_tab, -1*t_tab]
        fig.add_trace(
            go.Scatter(
                x=x, y= [0, t_student.pdf(0, n_size-1, loc=0, scale=1)],
                mode='lines', line = dict(color = 'blue', dash="dot", width=1),
                name=textos[4], legendgroup='2', showlegend=False,
                customdata= np.round(x, dec_places),
                hovertemplate = textos[4] + " = %{customdata} <extra></extra>"
            )
        )
        # adding t calc
        x = [t_calc, t_calc]
        fig.add_trace(
            go.Scatter(
                x=x, y= [0, t_student.pdf(np.abs(t_calc), n_size-1, loc=0, scale=1)],
                mode='lines', line = dict(color = 'red', dash="dot", width=2),
                name=textos[3], legendgroup='3', showlegend=True,
                customdata= np.round(x, dec_places),
                hovertemplate =  textos[3] + " = %{customdata} <extra></extra>"
            )
        )
        fig.update_layout(
                        legend = dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                        )
                    )

    fig.add_hline(y=0, line_width=2, line_dash="solid", line_color="black")

    # ajustando o layout
    fig.update_layout(
                    template='simple_white',
                    hoverlabel = dict(
                        font_size = 16,
                        font_family = "Rockwell"
                    ),
                    legend = dict(
                        font_size = 12,
                        font_family = "Rockwell",
                        bordercolor="Black", borderwidth=1
                    ),
                    margin={"r":0,"l":0,"b":0, 't':30}, # removendo margens desnecessárias
                )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, title=None)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, title=None)


    return fig, table, result





### ------ ------ ###
@callback(
        Output('mean-with-constant-title', 'children'),
        Output('mean-with-constant-text-significance-level', 'children'),
        Output('mean-with-constant-text-norm-test', 'children'),
        Output('mean-with-constant-text-outlier-test', 'children'),
        Output('mean-with-constant-text-decimal-places', 'children'),
        Output('mean-with-constant-offcanvas', 'title'),
        Output('mean-with-constant-text-data-summary', 'children'),
        Output('mean-with-constant-text-test-result', 'children'),
        Output('mean-with-constant-text-student-plot', 'children'),
        Output('mean-with-constant-text-result', 'children'),
        Output('mean-with-constant-text-test-type', 'children'),
        Output('mean-with-constant-text-known-value', 'children'),
        Output('mean-with-constant-text-drag-drop', 'children'),
        Output('mean-with-constant-offcanvas-test-summary', 'children'),
        Output('mean-with-constant-download-button', 'children'),

        Input('language-picker', 'value'),
              )
def language_picker(lang):
    if lang == "en":
        title = "Comparison of an average with a known value"
        sig_alfa = "Significance level"
        normality_test = "Normality Test"
        outliers_test = "Outliers Test"
        decimal_places = "Decimal places"
        preferences = "Preferences"
        data_summary = "Data summary"
        test_result = "Test result"
        t_plot = "Student's t distribution graph"
        conclusion = "Conclusion"
        test_type = "Test type"
        know_value = "Known value"
        drag_drop = "Drag and Drop or Select Files"
        test_summary = (
            "Use this Student test when you want to compare the *mean* of a data set (with **Normal** distribution) with a *determined or known* value.",
            "The test statistic is given by",
            "where $\\overline{x}$, $s_{x}$ and $n$ are the mean, sample standard deviation and dataset size, respectively, and $\\mu$ is the expected value. The test can be approached in two ways:",
            "Two-sided:",
            "One-sided:",
            "In both cases, we accept the null hypothesis when $p-value \\geq \\alpha$. Otherwise ($p-value < \\alpha$), we reject the null hypothesis."
        )
        download_button = "Download the example csv file"

    else: #"pt-br"
        title = "Comparação de uma média com um valor conhecido"
        sig_alfa = "Nível de significância"
        normality_test = "Teste de Normalidade"
        outliers_test = "Teste de Outliers"
        decimal_places = "Casas decimais"
        preferences = "Preferências"
        data_summary = "Resumo dos dados"
        test_result = "Resultado do teste"
        t_plot = "Gráfico da distribuição t de Student"
        conclusion = "Conclusão"
        test_type = "Tipo de teste"
        know_value = "Valor conhecido"
        drag_drop = "Arraste e solte ou selecione arquivos"
        test_summary = (
            "Utilize este teste de Student quando quiser comparar a *média* de um conjunto de dados (com distribuição **Normal**) com um valor *determinado ou conhecido*.",
            "A estatística do teste é dada por",
            "onde $\\overline{x}$, $s_{x}$ e $n$ são a média, o desvio padrão amostral e o tamanho do conjunto de dados, respectivamente, e $\\mu$ é o valor esperado. O teste pode ser abordado de duas formas:",
            "Bilateral:",
            "Unilateral:",
            "Nos dois casos, aceitamos a hipótese nula quando $p-valor \\geq \\alpha$. Caso contrário ($p-valor < \\alpha$), rejeitamos a hipótese nula."
        )
        download_button = "Baixe o arquivo csv de exemplo"



    test_resume = html.Div([
            html.P([
                dcc.Markdown(test_summary[0]),
                html.Span(" "),
                html.Span(test_summary[1]),
                html.Span(":"),
            ]),
            dcc.Markdown("$t_{calc}=\\frac{\\overline{x}-\\mu}{s_{x}/\\sqrt(n)}$", style={"textAlign": "center"}, mathjax=True),
            dcc.Markdown(test_summary[2], mathjax=True),
            html.Ul([
                html.Li(test_summary[3]),
            ]),
            dcc.Markdown("$H_{0}: \\overline{x} = \\mu$", style={"textAlign": "center"}, mathjax=True),
            dcc.Markdown("$H_{1}: \\overline{x} \\neq \\mu$", style={"textAlign": "center"}, mathjax=True),
            html.Ul([
                html.Li(test_summary[4]),
            ]),
            dcc.Markdown("$H_{0}: \\overline{x} = \\mu$", style={"textAlign": "center"}, mathjax=True),
            dcc.Markdown("$H_{1}: \\overline{x} < \\mu$", style={"textAlign": "center"}, mathjax=True),
            dcc.Markdown("$H_{1}: \\overline{x} > \\mu$", style={"textAlign": "center"}, mathjax=True),
        dcc.Markdown(test_summary[5], mathjax=True),
        ], style={"textAlign": "justify"})




    return (title, sig_alfa, normality_test, outliers_test, decimal_places, preferences, data_summary, test_result, t_plot, conclusion,
                test_type, know_value, drag_drop, test_resume, download_button)




@callback(
    Output("mean-with-constant-download-csv", "data"),
    Input("mean-with-constant-download-button", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    df = pd.DataFrame({
        'amostras' : (1, 2, 3, 4, 5, 6, 7, 8, 9),
        'rotulo' : (767.8, 764.1, 716.8, 750.2, 756.0, 692.5, 736.1, 746.1, 731.4)
    })
    return dcc.send_data_frame(df.to_csv, "example.csv", index=False)



#################################################
### Calback para abrir a aba de configurações ###
#################################################
@callback(
    Output("mean-with-constant-offcanvas", "is_open"),
    Input("mean-with-constant-open-offcanvas", "n_clicks"),
    [State("mean-with-constant-offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open



#
