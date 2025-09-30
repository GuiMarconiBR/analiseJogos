#importando as bibliotecas
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np
import plotly.express as px
import plotly.io as pio
sns.set_palette("deep")
sns.set_style("darkgrid")

#carregando o dataset
df = pd.read_csv('./dados/vgsales(in).csv', sep=';')

# Traduzindo as colunas para PT-BR
df.columns = ['posicao de venda', 'jogo', 'console', 'ano de lancamento', 'genero', 'publicadora', 'vendas américa do norte', 'vendas europa', 'vendas japao', 'vendas outras regioes', 'vendas totais']

# Removendo dados sem que não tenham o ano de lançamento informado
df.dropna(subset=['ano de lancamento'], inplace = True)

# Registrando como 'Desconhecida' publicadoras faltantes no dataset
df['publicadora'] = df['publicadora'].fillna('desconhecida')

# transformando Ano de Lançamento em Int
df['ano de lancamento'] = df['ano de lancamento'].astype(int)


# GRAFICOS de barras horizontal com os dez jogos mais vendidos EM MILHOES
# top_10 = df.sort_values('vendas totais', ascending=False).head(10)
# plt.figure(figsize=(10,7))
# sns.barplot(x='vendas totais', y='jogo', data=top_10)
# plt.ylabel('Jogos', size=13)
# plt.xlabel('Total de vendas (em milhões)', size=13)
# plt.title('Os dez jogos mais vendidos entre 1980 e 2016', size=15)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.show()

# GRAFICO: Consoles com mais lançamentos (top 10)
# most_releases = df['console'].value_counts().head(10).reset_index()
# most_releases.columns = ['console', 'lançamentos']
# plt.figure(figsize=(10,7))
# sns.barplot(x='lançamentos', y='console', data=most_releases)
# plt.xlabel('Número de lançamentos', size=13)
# plt.ylabel('Consoles', size=13)
# plt.title('Consoles com maior número de lançamentos', size=15)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()


# GRAFICOS Consoles que mais venderam jogos EM MILHOES
# most_sales = pd.DataFrame(df.groupby('console')[['vendas totais']].sum().sort_values('vendas totais', ascending = False).head(10))
# plt.figure(figsize=(10,7))
# sns.barplot(x = most_sales.index, y = 'vendas totais', data = most_sales)
# plt.ylabel('Jogos vendidos (em milhões)', size = 13)
# plt.xlabel('consoles', size = 13)
# plt.title('Consoles que mais venderam jogos', size = 15)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12, rotation=80)
# plt.show()

#GRAFICO frequencia anual de lancamento
# plt.figure(figsize=(10,7))
# sns.histplot(data=df, x='ano de lancamento', bins=35, kde=True)
# plt.ylabel('Lançamentos', size=13)
# plt.xlabel('Ano', size=13)
# plt.title('Frequência anual de lançamentos', size=15)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12, rotation=80)
# plt.show()

# GRAFICO DE VENDAS POR GENERO PIRÂMIDE
genres = df.groupby('genero')['vendas totais'].sum().sort_values(ascending=False).reset_index()
fig = px.funnel(genres, y='genero', x='vendas totais')

fig.update_layout(
    title={
        'text': "pirâmide de vendas por gênero (em milhões)",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()

