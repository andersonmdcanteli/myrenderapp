
### ------ Imports ------ ###

### ------ Dash ------ ###
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

### ------ Native ------ ###
from random import randint, shuffle


### ------ making the page layout ------ ###
layout = html.Div([
    # first row
    dbc.Row([
        dbc.Col(
            html.H1("Esta página NÃO EXISTE!!! 😱"),
            style={"textAlign": "center"}
        ),
    ], className="title_spacing"),
    # second row
    dbc.Row([
        dbc.Col([
            html.H1([
                "Ou será que ela EXISTE???",
                "🤔"
            ]),
        ],style={"textAlign": "center"}
        ),
    ], className="title_spacing"),
    # info
    dbc.Row([
        dbc.Col([
            html.P([
                "Existe 50% de chance desta página ",
                html.Span("existir", style={"fontStyle": "italic"}),
                " e 50% de chance dela ",
                html.Span("não existir", style={"fontStyle": "italic"}),
                ". Até que um experimento seja feito, não é possível saber o estado desta página..."
            ])
        ], lg=8, width="auto")
    ], justify="center", className="title_spacing"),
    # info
    dbc.Row([
        dbc.Col([
            html.P([
                "Clique no ",
                html.A("Octocat", href="https://en.wikipedia.org/wiki/Octocat"),
                " para realizar um experimento e descobrir o estado da página!",
            ])
        ], lg=8, width="auto", )
    ], justify="center"),
    # Row to add the button that must be replaced be a figure
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Button(
                        html.Img(src="assets/schrodinger/schrodinger.png", alt="schrodinger github's cat",
                        style={"maxWidth": "75%", "height": "auto"}),
                        id='not-found-n-clicks',
                        n_clicks=0,
                        style={"backgroundColor": "rgba(0,0,0,0)", "border": "0px rgba(0,0,0,0)"}
                    ),
            ])
        ], lg=3, width=6),
    ], justify="center", className="title_spacing", style={"textAlign": "center"}),
    dbc.Row([
        dbc.Col(
            html.Div(id='container-not-found-result',
                     children=''),
                     lg=8, width="auto"
        ),
    ],justify="center"),
    dbc.Row([
        dbc.Col(
            html.Div(id='container-not-found-state',
                     children=''),
                     lg=8, width="auto"
        ),
    ],justify="center", style={"paddingBottom": "40px", "paddingTop": "20px", "textAlign": "center"}),
    html.Div(id="gatos-vivo-morto", ),
    # info
    dbc.Row(
        dbc.Col(
            html.H3("Detalhes do experimento"), lg=8, width="auto"
        ), justify="center", style={"paddingTop": "30px"}
    ),
    dbc.Row(
        dbc.Col(
            html.P([
                "Esta é uma página que ",
                html.Span("simula", style={"fontStyle": "italic"}),
                " o experimento de Schrodinger, o que é feito da seguinte forma: a página (que corresponde ao gato) foi armazenada em um repositório fechado (que depois de fechado não tem contato com o mundo exterior), e, nesta página, existe um algoritmo que escolhe um único número entre os números 0 e 1. ",
                "Se o resultado do sorteio for o número 0, a página será considerada como existente (e assim o gato está VIVO). Se o sorteio retornar o número 1, a página será considerada inexistente (e o gato esta MORTO).",
            ]), lg=8, width="auto"
        ), justify="center"
    ),
    dbc.Row(
        dbc.Col(
            html.P("Este algoritmo simula todo o aparato proposto por Schrodinger, que foi pensado da seguinte forma:"), lg=8, width="auto"
        ), justify="center"
    ),
    dbc.Row(
        dbc.Col(
            html.Ul([
                html.Li("Um frasco com material radioativo, que tem exatamente 50% de chance de decair todas as vezes que o experimento for realizado;"),
                html.Li("Um medidor Geiger para verificar se o material radioativo decai;"),
                html.Li("Um frasco contendo uma quantidade suficiente de cianeto para matar o gato;"),
                html.Li("Um martelo que somente será acionado se o medidor Geiger identificar o material radioativo. Se isso acontecer, o martelo quebrará o frasco que contém com cianeto, o que matará o gato."),
            ]), lg=8, width="auto"
        ), justify="center"
    ),
    dbc.Row(
        dbc.Col(
            html.P('Todo este sistema era colocado em um recipente completamente isolado do mundo externo (como um "cofre"), e seria mantido assim por 30 minutos, que é o tempo exato que o material radioativo teria 50% de chance de decair.'),
            lg=8, width="auto"
        ), justify="center"
    ),
    dbc.Row(
        dbc.Col(
            html.P('O "problema" é que até que um experimento terminar, não é possível saber se esta página existe ou se não existe (se o gato vive ou não). No momento em que o experimento termina (que simula o observador abrindo a caixa para verificar o status do gato), a função aleatória será chamada e o resultado será exibido (o sistema colapsa para estado vivo ou morto), o que irá reduzir a página a um estado de existente ou não existente.'),
            lg=8, width="auto"
        ), justify="center"
    ),
    dbc.Row(
        dbc.Col(
            html.P('Por este motivo, esta é uma página quântica 😛'),
            lg=8, width="auto"
        ), justify="center"
    ),
    dbc.Row(
        dbc.Col(
            html.Hr(), lg=10, width="auto"
        ), justify="center", className="title_spacing"
    ),
    dbc.Row(
        dbc.Col([
            html.P([
                "O conteúdo desta página é inspirado no experimento teórico de Schoridnger, ",
                html.Strong("mas esta longe de estar preciso!"),
                " Se você quiser aprender mais, você pode ler sobre este experimento, clicando ",
                html.A("aqui", href="https://en.wikipedia.org/wiki/Schr%C3%B6dinger%27s_cat#:~:text=Schr%C3%B6dinger's%20cat%20is%20a%20thought,Copenhagen%20interpretation%20of%20quantum%20mechanics."),
                ", ou você pode assistir o professor Martyn Poliakoff falando sobre este experimento logo aqui embaixo, ou clicando ",
                html.A("aqui", href="https://www.youtube.com/watch?v=CrxqTtiWxs4"),
                "."
            ])
        ], lg=8, width="auto"
        ), justify="center", className="title_spacing"
    ),
    dbc.Row(
        dbc.Col(
            html.Iframe(width="560", height="315", src="https://www.youtube.com/embed/CrxqTtiWxs4", title="YouTube video player",
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture")
        ), justify="center", style={"textAlign": "center"}
    ),
    # empty row to insert the cats current life
    dbc.Row([
        dbc.Col(
            8, id="not-found-current-lifes", style={'display': 'none'}
        )
    ]),
    # empty div to store the total number of the cat life
    dcc.Store(id='not-found-store-n-lifes'),
    # empty div to trigger the total number of the cat life
    html.Div(id='not-found-respawn',children=[],style={'display': 'none'}),
    # modal
    html.Div(id="not-found-modal")

], id="not-found-div")




### ------ Calback to store how many cats can be used on the experiment ------ ###
@callback(
    Output(component_id='not-found-store-n-lifes', component_property='data'),
    [Input(component_id='not-found-respawn', component_property='value')]
)
def update_respawn(cats_life):
    return 8





### ------ Callback to update cats data ------ ###
@callback(
    [
    Output(component_id='container-not-found-result', component_property='children'),
    Output(component_id='container-not-found-state', component_property='children'),
    Output(component_id='not-found-current-lifes', component_property='children'),
    Output(component_id="gatos-vivo-morto", component_property='children'),
    Output(component_id="not-found-modal", component_property="children"),
    Output(component_id='not-found-div', component_property='className'),
    ],
    [
    Input(component_id='not-found-n-clicks', component_property='n_clicks'),
    Input(component_id='not-found-store-n-lifes', component_property='data'),
    Input(component_id="not-found-current-lifes", component_property='children')
    ],
)
def update_output(n_clicks, lifes, current_lifes):
    # prevent update when the cat is dead
    if current_lifes == 0:
        raise PreventUpdate
    # if no clicks, show nothings
    if n_clicks == 0:
        img = "assets/schrodinger/schrodinger.png"
        alt = "schrodinger"
        estado = html.P("")
        resultado = html.P("")
        current_lifes = lifes
        modal_trigger = False
    else:
        # fazendo o experimento
        experimento = randint(0,1)
        # triggering the modal
        modal_trigger = True
        # if the cat is is ALIVE
        if experimento == 0:
            estado = html.P(
                f"Felizmente, o sensor Geiger não mediu nenhuma degradação do átomo, o que não disparou o martelo e manteve o cianeto intacto dentro do frasco. Dessa forma, o Octocat segue com suas {current_lifes} vidas!"
            )
            resultado = html.H3(
                "Então, esta página EXISTE!"
            )
            img = "assets/schrodinger/schrodinger_vivo.png"
            alt = "schrodinger github's cat alive"
        else:
            # if the cat is DEAD
            current_lifes -= 1
            estado = html.P(
                "Infelizmente, o sensor Geiger mediu o decaimento de um átomo. Dessa forma, o martelo foi disparado, quebrando o frasco que continha o cianeto, o que fez com o Octocat perdesse uma de suas vidas..."
            )
            resultado = html.H3(
                "Infelizmente, esta página NÃO EXISTE!"
            )
            img = "assets/schrodinger/schrodinger_morto.png"
            alt = "schrodinger github's cat dead"

    # inserting the cats faces
    colunas = []
    if n_clicks == 0:
        linha=[]
    else:
        # making the rows to place the dead and alive cats
        colunas = []
        for i in range(lifes - current_lifes):
            colunas.append(
                dbc.Col(
                    html.Img(src="assets/schrodinger/schrodinger_morto.png", alt="schrodinger github's cat dead",
                    style={"maxWidth": "75%", "height": "auto"}), lg=2, xs=3
                )
            )

        if experimento == 0:
            colunas.append(
                dbc.Col(
                    html.Img(src="assets/schrodinger/schrodinger_vivo.png", alt="schrodinger github's cat alive",
                    style={"maxWidth": "75%", "height": "auto"}), lg=2, xs=3
                )
            )

        if len(colunas) < 5:
            linha = [dbc.Row(colunas, style={"textAlign": "center"}, justify="center")]
        else:
            linha = [
                dbc.Row(colunas[:4], style={"textAlign": "center"}, justify="center"),
                dbc.Row(colunas[4:], style={"textAlign": "center"}, justify="center"),
                ]





    # building the modal
    if current_lifes > 0:
        # if the cat is ALIVE
        modal_body = html.Div([
            dbc.Row(
                dbc.Col(estado)
            ),
            dbc.Row(
                dbc.Col(resultado, style={"textAlign": "center"})
            ),
            dbc.Row(
                dbc.Col(
                    html.Img(
                        src=img, alt=alt,
                        style={"maxWidth": "50%", "height": "auto"}
                    ), width="auto", style={"textAlign": "center"}
                ), justify="center",
            )
        ])
        update_style = ""
    else:
        # changes the className to fadeout after some time :D
        update_style = "hideMe"
        # if the cat is DEAD
        modal_body = html.Div([
            dbc.Row(
                dbc.Col(
                    f"Você conduziu o experimento por {n_clicks} vezes, fazendo com que o Octocat perdesse todas as suas {lifes} vidas."
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.Br()
                )
            ),
            dbc.Row(
                dbc.Col(
                    "Recarregue a página para que possamos utilizar um novo Octocat no experimento."
                )
            ),
        ])

    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Resultado do experimento"), close_button=True),
                    dbc.ModalBody(modal_body),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Fechar",
                            id="not-found-close-centered",
                            className="me-1",
                            color="secondary",
                            outline=True,
                            n_clicks=0,
                        )
                    ),
                ],
                id="not-found-modal-centered",
                centered=True,
                is_open=modal_trigger,
            ),
        ]
    )

    return estado, resultado, current_lifes, linha, modal, update_style






### ------ Callback to MODAL ------ ###
@callback(
    Output(component_id="not-found-modal-centered", component_property="is_open"),
    [Input(component_id="not-found-close-centered", component_property="n_clicks")],
    [State(component_id="not-found-modal-centered", component_property="is_open")],
)
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open






#
