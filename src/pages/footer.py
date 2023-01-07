### ------ Imports ------ ###
### --- dash --- ###
from dash import html, Input, Output, callback, dcc
import dash_bootstrap_components as dbc

### --- python --- ###
from random import choice
from datetime import date

### --- dataset --- ###
from datasets.frases import frases


### --- color name --- ###
text_color = "rgba(134,143,147,255)"

### ------ data for dropdown ------ ###
language_options = [
    {'label': 'pt-br', 'value': 'pt-br'},
    {'label': 'en', 'value': 'en'},
]


### --- footer --- ###
footer = html.Footer(
    children=[
        html.Div(
            [
            # empty div to trigger autor, frase and year
            html.Div(id='none',children=[],style={'display': 'none'}),

            dbc.Row([
                # first column
                dbc.Col([
                    # frase
                    dbc.Row(
                        dbc.Col(
                            html.P(
                                html.I(id="footer-frase"),
                                style={"textAlign": "center", "color": "rgba(134,143,147,255)"})
                        )
                    ),
                    # autor
                    dbc.Row(
                        dbc.Col(
                            html.P(
                                html.Strong(id="footer-autor"),
                                style={
                                    "textAlign": "right",
                                    "color": "rgba(134,143,147,255)",
                                    "paddingRight": "10%"
                                    })
                        )
                    )

                ], lg=5, align="center"),
                # second column
                dbc.Col([
                    dbc.Row([
                        dbc.Col(
                            html.P(
                                html.Strong(
                                    id="footer-contact-option"
                                ), style={"color": text_color, "textAlign": "center", "textDecoration": "underline", "paddingTop": "15px"}
                            )
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-telegram",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://t.me/AndersonCanteli"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-facebook-messenger",
                                    style={"color": text_color, }
                                    ),
                                ), href="http://m.me/profCanteli"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-whatsapp",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://wa.me/+5541996808012"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-solid fa-envelope",
                                    style={"color": text_color, }
                                    ),
                                ), href="mailto:andersonmdcanteli@gmail.com"
                            ), width="auto"
                        ),
                    ], justify="evenly"),
                    dbc.Row([
                        dbc.Col(
                            html.P(
                                html.Strong(
                                    id="footer-social-media-option"
                                ), style={"color": text_color, "textAlign": "center", "textDecoration": "underline", "paddingTop": "15px"}
                            )
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-facebook",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://www.facebook.com/profCanteli"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-instagram",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://www.instagram.com/andersonmdcanteli/"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-youtube",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://www.youtube.com/channel/UCf96X2h-DrdY4Oc615fZBew"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-linkedin",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://www.linkedin.com/in/anderson-canteli-7431624a/"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-solid fa-house",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://andersonmdcanteli.github.io/"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-researchgate",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://www.researchgate.net/profile/Anderson-Canteli-2"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-solid fa-building-columns",
                                    style={"color": text_color, }
                                    ),
                                ), href="http://lattes.cnpq.br/6961242234529344"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.I(className="fa-brands fa-github",
                                    style={"color": text_color, }
                                    ),
                                ), href="https://github.com/andersonmdcanteli"
                            ), width="auto"
                        ),
                    ], justify="evenly"),
                    dbc.Row([
                        dbc.Col(
                            html.P(
                                html.Strong(
                                    "Projetos"
                                ), style={"color": text_color, "textAlign": "center", "textDecoration": "underline", "paddingTop": "15px"}
                            )
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.Img(src='assets/cavs.png',
                                    style={"color": text_color, }
                                    ),
                                ), href="http://www.prppg.ufpr.br/site/posalim/pb/aplicativos/"
                            ), width="auto"
                        ),
                        dbc.Col(
                            html.A(
                                html.H3(
                                    html.Img(src='assets/pycafee.png',
                                    style={"color": text_color, }
                                    ),
                                ), href="https://pycafee.readthedocs.io/en/latest/"
                            ), width="auto"
                        ),

                    ], justify="center"),
                    dbc.Row([
                        dbc.Col(
                            html.P([
                                html.Strong(
                                    id="footer-language-option"
                                ),
                            ],
                            style={"color": text_color, "textAlign": "center", "textDecoration": "underline",}
                            )
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dcc.Dropdown(
                                id='language-picker',
                                value = language_options[0]['value'],
                                options = language_options,
                                clearable = False,
                                persistence=True,
                                persistence_type="local",
                                style={"color": text_color, "backgroundColor": "rgba(134,143,147,255)",  }
                                ),
                            style={"color": text_color, "textAlign": "center", "textDecoration": "underline",},
                            width=6
                        ),
                    ], justify="center"),

                ], lg=5, )
            ], justify="evenly"),
            dbc.Row(
                dbc.Col(
                    html.Hr(className="my-2", style={"backgroundColor": "rgba(134,143,147,255)"}),
                    lg=10,
                ), justify="center", style={"paddingBottom": "20px", "paddingTop": "20px"}
            ),
            dbc.Row(
                dbc.Col([
                    html.A(children =
                        [
                            html.Label("Anderson Canteli ("),
                            html.Label(id="footer-ano"),
                            html.Label(")"),
                        ],
                        href="/about"
                        ),

                ], className='text-center'),
            ),
         ], className="h-100 p-5", style={"backgroundColor": "rgba(40,45,50,255)"},
         ),
    ],
)



### ------ ------ ###
@callback(
        Output('footer-language-option', 'children'),
        Output('footer-social-media-option', 'children'),
        Output('footer-contact-option', 'children'),

        Input('language-picker', 'value'),
              )
def language_picker(lang):
    if lang == "en":
        return "Language", "Social Media", "Contact"
    else:
        return "Idioma", "Redes Sociais", "Contato"


### ------ Callback to update autor, frase and year on refresh ------ ###
@callback(
        Output('footer-autor', 'children'),
        Output('footer-frase', 'children'),
            Output('footer-ano', 'children'),
        [Input('none', 'children')]
    )
def get_autor(none):
    autor = choice(list(frases.frases.keys()))
    frase = frases.frases[autor]
    ano = date.today().year
    return autor, frase, ano



















#
