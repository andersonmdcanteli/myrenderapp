from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_extensions as de

url_loties = "https://assets6.lottiefiles.com/packages/lf20_gp8xcujl.json"
options_lotties = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))



layout = html.Div([
    dbc.Row(
        dbc.Col(
            html.H2("Home page"),
        ), style={"text-align": 'center'}, className="title_spacing",
    ),
    dbc.Row([
        dbc.Col([
            html.P([
                # "Esta aplicação faz parte de um projeto educacional de estatistica para pesquisadores, mas ela é apenas uma amostra de um dashboard interativo desenvolvido para treinamento de estatística básica."
            ], style={"textAlign": 'justify'}, className="my_spacing"),
        ]),
    ]),
    # dbc.Row([
    #     dbc.Col([
    #         html.P([
    #             "Clique ",
    #             html.A("aqui", href="/estatistica-basica"),
    #             " para acessar o conteúdo da página."
    #         ], style={"textAlign": 'justify'}, className="my_spacing"),
    #     ]),
    # ]),
    dbc.Row(
        dbc.Col(
            html.Div(de.Lottie(options=options_lotties, width="50%", height="50%", url=url_loties))
        )
    ),

])
