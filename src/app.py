from dash import Dash, dcc, html, Input, Output, callback
from pages import home, about, page_not_found, page_mean_with_constant
from pages import footer
from datetime import date
import dash_bootstrap_components as dbc
from datasets import frases


FA = "https://use.fontawesome.com/releases/v6.0.0/css/all.css"

external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.themes.GRID, FA]

app = Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets = external_stylesheets,
                # external_scripts = mathjax,
                meta_tags = [{
                        'name': 'viewport',
                        'content': 'width=device_width',
                        'initial-scale': 1.0
                }])

server = app.server

## Navbar Simples
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Mean with constant", href="/mean-constant")),
        dbc.NavItem(dbc.NavLink(
                        html.I(className="fas fa-mug-hot", style={"color": 'yellow'}),
                        href="/about")),
            ],
    brand="Home",
    brand_href="/",
    color="darkgreen",
    dark=True,
)



## Criando o layout
app.layout = html.Div([
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(
        id='page-content',
        style={'margin-right': '2.5%', 'margin-left': '2.5%', 'margin-top': '10px', }
        ),
    footer.footer,
], style={"backgroundColor": "ivory"})


@callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == "/mean-constant":
        return page_mean_with_constant.layout
    elif pathname == '/home':
        return home.layout
    elif pathname == '/about':
        return about.layout
    else:
        return page_not_found.layout

if __name__ == '__main__':
    app.run_server(debug=True)
