
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc

from datetime import date








sobre_mim = html.Div([
    html.P(children=[
        "Olá! Eu sou o Anderson Canteli, o criador deste interativo site! Atualmente tenho ",
        html.Span(id="about-birth-year"),
        " anos, moro em Curitiba-PR e trabalho com consultoria e treinamento em delineamentos experimentais, além de me aventurar como professor aulas de Python, criação de dashboards e análise de dados.",
    ]),
    html.P([
        "Sou bacharel (UEM), mestre (UFPR) e doutor (UFPR) em Engenharia de Alimentos. Já trabalhei com desenvolvimento de produtos alimentícios, ",
        "com controle de qualidade em laticínos, mas me especializei em processos de separação, especialmente o processo de adsorção. ",
        "Foi neste tema que desenvolvi meu mestrado, onde utilizei o processo de adsorção em leito fixo para estudar a ",
        html.A("Recuperação do aroma de café, benzaldeído, em coluna de adsorção utilizando carvão ativado (2013)", href="https://acervodigital.ufpr.br/handle/1884/30536"),
    ]),
    html.P([
        "No doutorado segui trabalhando com adsorção, onde desenvolvi o trabalho ",
        html.A("Adsorção de corante por um biossorvente obtido do casulo do bicho-da-seda (Bombyx mori): experimentos e modelagem ﻿(2018)", href="https://acervodigital.ufpr.br/handle/1884/58189"),
        ", onde além de abordar todas as etapas do processo de adsorção (desenvolvimento do material, caracterização, adsorção e dessorção em batelada, adsorção e dessorção em leito fixo, além de ciclos de adsorção), ",
        "também me desafiei a ir além na modelagem dos dados, tendo desenvolvido uma nova equação para o processo de dessorção em batelada, além de um software (desenvolvido em Matlab, chamando de CAVS - adsorption evaluation) para realizar os ajustes dos modelos, bem como avaliar os resultados de forma adequada. ",
        "Ainda durante o desenvolvimento da tese, me especializei em delineamentos experimentais (DOE), uma das técnicas mais importantes para otimização de produtos e processos."
    ]),
    html.P([
        "Após a tese comecei meus estudos de Python, voltado especialmente para análise de dados. Contudo, o primeiro grande resultado foi o desenvolvimento do software CAVS 2.0, que foi um repaginação do software que havia sido desenvolvido em Matlab. Esta nova versão foi registrada no INPI, o que me rendeu o título de Inventor (mais detalhes sobre o software abaixo). ",
        "Atualmente, sigo trabalhando com análise de dados, especialmente delineamentos experimentais, e também com Python, que utilizo para fazer análises mais robustas quando necessário e também dashboards, como este site, ",
        "além de canal do YouTube, onde adiciono tutoriais dos principais softwares de análises de dados que utilizei durante mestrado e doutorado. "
        "Também sou o mantenedor do CAVS, tanto de atualizações como das redes sociais e tutoriais. ",
        "Mais recentemente, iniciei um novo projeto como desenvolvedor, o PyCafee, que é uma biblioteca em Python voltada para pesquisadores (mais detalhes abaixo). ",
        "As vezes me aventuro como palestrante, onde já falei sobre criaividade como ferramenta para solução de problemas, sobre o por que que pesquisadores deveriam saber programação, e também sobre ",
        html.A("adsorção", href="https://youtu.be/roxJW0tW9dk?t=369"),
        "."
    ])
])

cavs = html.Div([
    html.P([
        "O ",
        html.Span("CAVS - adsorption evalution", style={"fontStyle": "italic"}),
        " é um software desenvolvido para auxiliar pesquisadores da área de adsorção de todo o mundo, que traz uma forma bastante simples e rápida de realizar análises de regressão dos principais modelos utilizados. "
        "Através do CAVS é possível realizar estudos sobre o processo de adsorção e dessorção, tanto em batelada como em leito fixo, cobrindo a maior parte dos modelos disponíveis na literatura (total de 27 modelos), além de realizar cálculos termodinâmicos do fenômeno e estimar parâmetros de leito fixo. ",
        "Além disso, as regressões utilizam as equações na sua forma não linear, o que evita os problemas relacionados a linearização dos dados, o que é prejudicial para a correta análise dos resultados experimentais."
    ]),
    html.P([
        "A ideia de desenvolver o software surge durante o desenvolvimento da minha dissertação de Mestrado, onde tive muitas dificuldades em realizar as modelagens devido a inconsistências da literatura no uso das equações, que podiam ser utilizadas na sua forma linear ou não linear, além de existir insconsitências entre referências sobre a forma das equações. ",
        "O CAVS resolve este problema: nele os principais modelos utilizados na literatura estão implementados com todas as equações na sua forma não linear, como uma rotina otimizada para a obtenção de resultados mais robustos. "
        u"Além disto, no campo de adsorção é muito comum utilizar apenas o coeficiente de determinação (R\u00B2) para determinar a ",
        html.Span("melhor representação dos dados experimentais", style={"fontStyle": "italic"}),
        ", que não é uma medida suficientemente rígida para ser utilizada como única medida. O CAVS também resolve este problema, fornecendo 5 testes estatísticos para verificar a qualidade do ajuste, além de gráficos de resíduos e as principais medidas de qualidade do ajuste. ",
        "Com todas as facilidades, o pesquisador que utiliza o ",
        html.Span("CAVS", style={"fontStyle": "italic"}),
        " tem mais tempo para obter dados de melhor qualidade, pois não precisa utilizar tempo para aprender a realizar as regressões, verificar se as equações estão corretas, e fazer todos os ajustes necessários. "
    ]),
    html.P([
        "Existem duas versões do CAVS: o CAVS 1.0, que foi desenvolvido utilizando Matlab (que é a versão contida na minha tese), mas que foi descontinuada e não esta disponível para download em nenhum lugar; "
        " e o CAVS 2.01, que foi desenvolvida em Python e que esta disponível para download em ",
        html.A("www.prppg.ufpr.br", href="http://www.prppg.ufpr.br/site/posalim/en/aplicativos/"),
        " para Windowns 7 64 bits ou superior. Esta versão esta registrada no INPI (",
        html.A("BR 51 2019 001440-5", href="https://transparencia.cc/dados/inpi/software/5120190014405-cavs-adsorption-evaluation/"),
        ") e seu uso é gratuito para toda a comunidade acadêmica. Tutoriais de como baixar, instalar e utilizar o software estão disponíveis no ",
        html.A("YouTube", href="https://www.youtube.com/channel/UCyby_B-VvUBJRcb_TF3l1wA"),
        ". Atualmente, o software tem mais de 900 downloads pelo mundo, sendo utilizado por pesquisadores na Inglaterra, Colômbia, Paquistão, Quênia, Ucrânia, Egito, Iraque, India, Japão, entre outros."
    ]),
    html.P([
        "Você encontra mais detalhes sobre o CAVS ",
        html.A("neste link", href="https://andersonmdcanteli.github.io/CAVS/"),
        "."
    ])
])

pycafee = html.Div([
    html.P(children=[
        "O PyCafee é uma biblioteca de análise de dados voltada para pesquisadores das áreas de ciências agrárias, biológicas e exatas que possuem pouca ou nenhuma experiência com programação. ",
    ]),
    html.P(children=[
        "Desenvolvida em Python, o projeto open source pode ser baixado utilizando o comando ",
        html.Span("pip install pycafee", style={"fontStyle": "italic"}),
        ", que irá instalar todas as dependências de forma bastante simplificada. Para deixar o projeto mais acessível, ",
        "o projeto utiliza as versões das dependências iguais as disponíveis no ambiente de colaboração da Google, o ",
        html.A("Google Colab", href="https://colab.research.google.com/"),
        ", o que permite utilizar todas as funcionalidades da biblioteca sem a necessidade fazer nenhuma instalação no seu computador.",
        " É necessário apenas ter uma conta da Google e conecção com a internet. Você encontra maiores detalhes em relação as requisitos necessários em ",
        html.A("pypi.org/project/pycafee", href="https://pypi.org/project/pycafee/"),
        "."
    ]),
    html.P([
        "Apesar de estar nos seus primeiros passos, a biblioteca já conta com funções para verificar Normalidade de dados, através de métodos gráficos e testes formais, ",
        "testes de outliers, comparação de amostras, entre outros. Além disso, a biblioteca conta com um sistema de banco de dados que permite alternar entre o idioma inglês e o português, facilitando ainda mais a vida do pesquisador.",
        " Também esta sendo feito um esforço para que a biblioteca esteja sempre bem documentada (",
        html.A("pycafee.readthedocs.io", href="https://pycafee.readthedocs.io/en/latest/index.html"),
        "), o que é mais uma fonte de informações para o pesquisador. Novas funções são adicionadas periodicamente! Todo o código é aberto, e pode ser acessado no respositório do github (",
        html.A("github.com/pycafee", href="https://github.com/pycafee"),
        ")."
    ]),
    html.P([
        "Acompanhe o nosso instagram ",
        html.A("@pycafee", href="https://www.instagram.com/pycafee/"),
        " para ficar atualizado com as novas funcionalidades e os rumos do projeto."
    ])
])




layout = html.Div([
    # empty div to trigger birth year
    html.Div(id='birth-year',children=[],style={'display': 'none'}),
    dbc.Row([
        dbc.Col(
            html.Hr(style={"height": "5px", "color": "darkgreen" }), width=5
        ),
        dbc.Col(
            html.H1(
            html.I(className="fa-solid fa-mug-saucer", style={"color": 'darkgreen'}),
            ), width=2, style={"textAlign": "center",}
        ),
        dbc.Col(
            html.Hr(style={"height": "5px", "color": "darkgreen" }), width=5
        ),
    ], justify='evenly', align="center", className="title_spacing"),
    # about
    dbc.Row([
        dbc.Col(
            html.H1("FOTO"), lg=4, style={"textAlign": "center"},
        ),
        dbc.Col([
            dbc.Row(
                dbc.Col(
                    html.H5("...")
                )
            ),
            dbc.Row(
                dbc.Col(
                    sobre_mim
                ), style={"textAlign": "justify"}
            )
        ], lg=6)
    ], align="center", justify="around"),
    # Cavs
    dbc.Row([
        dbc.Col([
            dbc.Row(
                dbc.Col(
                    html.H5(children=[
                        html.Strong("CAVS - adsorption evaluation", style={"color": "darkgreen"}),
                    ])
                )
            ),
            dbc.Row(
                dbc.Col(
                    cavs, style={"textAlign": "justify"}
                )
            ),

        ], lg=6),
        dbc.Col([
            dbc.Row(
                dbc.Col(
                    html.Img(src="assets/cavs_full.png", alt="Pycafee logo")
                ), style={"textAlign": "justify"}
            )
        ], lg=4)
    ], align="center", justify="around"),
    # Pycafee
    dbc.Row([
        dbc.Col(
            html.Img(src="assets/Safeimagekit-resized-imgpng.png", alt="Pycafee logo"),
            lg=4, style={"textAlign": "center"},
        ),
        dbc.Col([
            dbc.Row(
                dbc.Col(
                    html.H5([
                        "Projeto ",
                        html.Strong("Py", style={"color": "rgba(249,204,83,255)"}),
                        html.Strong("Cafee", style={"color": "rgba(193,1,0,255)"}),
                    ])
                )
            ),
            dbc.Row(
                dbc.Col(
                    pycafee
                ), style={"textAlign": "justify"}
            )
        ], lg=6)
    ], align="center", justify="around"),
])



### ------ Callback to update autor, frase and year on refresh ------ ###
@callback(
        Output('about-birth-year', 'children'),
        [Input('birth-year', 'children')]
    )
def calculate_age(none):
    ano, mes, dia = 1988, 7, 22
    today = date.today()
    return today.year - ano - ((today.month, today.day) < (mes, dia))










# asd
