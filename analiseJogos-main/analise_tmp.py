#importando as bibliotecas
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np
import plotly.express as px
import plotly.io as pio
sns.set_palette("deep")
sns.set_style("darkgrid")

#carregando o dataset
df = pd.read_csv('./dados/vgsales(in).csv', sep=';')

# Traduzindo as colunas para PT-BR
df.columns = ['posicao de venda', 'jogo', 'console', 'ano de lancamento', 'genero', 'publicadora', 'vendas amÃ©rica do norte', 'vendas europa', 'vendas japao', 'vendas outras regioes', 'vendas totais']

# Removendo dados sem que nÃ£o tenham o ano de lanÃ§amento informado
df.dropna(subset=['ano de lancamento'], inplace = True)

# Registrando como 'Desconhecida' publicadoras faltantes no dataset
df['publicadora'] = df['publicadora'].fillna('desconhecida')

# transformando Ano de LanÃ§amento em Int
df['ano de lancamento'] = df['ano de lancamento'].astype(int)


# GRAFICOS de barras horizontal com os dez jogos mais vendidos EM MILHOES
# top_10 = df.sort_values('vendas totais', ascending=False).head(10)
# plt.figure(figsize=(10,7))
# sns.barplot(x='vendas totais', y='jogo', data=top_10)
# plt.ylabel('Jogos', size=13)
# plt.xlabel('Total de vendas (em milhÃµes)', size=13)
# plt.title('Os dez jogos mais vendidos entre 1980 e 2016', size=15)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.show()

# GRAFICO: Consoles com mais lanÃ§amentos (top 10)
# most_releases = df['console'].value_counts().head(10).reset_index()
# most_releases.columns = ['console', 'lanÃ§amentos']
# plt.figure(figsize=(10,7))
# sns.barplot(x='lanÃ§amentos', y='console', data=most_releases)
# plt.xlabel('NÃºmero de lanÃ§amentos', size=13)
# plt.ylabel('Consoles', size=13)
# plt.title('Consoles com maior nÃºmero de lanÃ§amentos', size=15)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()


# GRAFICOS Consoles que mais venderam jogos EM MILHOES
# most_sales = pd.DataFrame(df.groupby('console')[['vendas totais']].sum().sort_values('vendas totais', ascending = False).head(10))
# plt.figure(figsize=(10,7))
# sns.barplot(x = most_sales.index, y = 'vendas totais', data = most_sales)
# plt.ylabel('Jogos vendidos (em milhÃµes)', size = 13)
# plt.xlabel('consoles', size = 13)
# plt.title('Consoles que mais venderam jogos', size = 15)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12, rotation=80)
# plt.show()

#GRAFICO frequencia anual de lancamento
# plt.figure(figsize=(10,7))
# sns.histplot(data=df, x='ano de lancamento', bins=35, kde=True)
# plt.ylabel('LanÃ§amentos', size=13)
# plt.xlabel('Ano', size=13)
# plt.title('FrequÃªncia anual de lanÃ§amentos', size=15)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12, rotation=80)
# plt.show()

# GRAFICO DE VENDAS POR GENERO PIRÃ‚MIDE
# genres = df.groupby('genero')['vendas totais'].sum().sort_values(ascending=False).reset_index()
# fig = px.funnel(genres, y='genero', x='vendas totais')

# fig.update_layout(
#     title={
#         'text': "pirÃ¢mide de vendas por gÃªnero (em milhÃµes)",
#         'y': 0.95,
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'
#     }
# )
# fig.show()

# ------------------ INÃCIO: BLOCO KNN ------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# 1) Features e alvo
X = df[['vendas amÃ©rica do norte', 'vendas europa', 'vendas japao', 'vendas outras regioes']].copy()
y = df['genero'].copy()

# 2) Limpeza
mask_valid = y.notna() & (X.sum(axis=1) > 0)
X = X[mask_valid]
y = y[mask_valid]

# 3) CodificaÃ§Ã£o dos gÃªneros
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4) Treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# 5) Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6) Modelo KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train_scaled, y_train)

# 7) PrediÃ§Ã£o e avaliaÃ§Ã£o
y_pred = knn.predict(X_test_scaled)

print("\n=== RESULTADOS KNN (vendas por regiÃ£o -> prever gÃªnero) ===")
print("NÃºmero de amostras (treino):", X_train.shape[0], "| (teste):", X_test.shape[0])
print("AcurÃ¡cia no conjunto de teste: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# 8) RelatÃ³rio em portuguÃªs
relatorio = classification_report(
    y_test, y_pred, target_names=le.classes_, output_dict=True
)
df_relatorio = pd.DataFrame(relatorio).transpose()
df_relatorio.rename(columns={
    'precision': 'precisao',
    'recall': 'revocacao',
    'f1-score': 'f1_score',
    'support': 'suporte'
}, inplace=True)

print("\nRelatÃ³rio de classificaÃ§Ã£o (em portuguÃªs):\n")
print(df_relatorio.round(2))

# 9) Matriz de confusÃ£o com nomes dos gÃªneros
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

print("\nMatriz de ConfusÃ£o (linhas = verdadeiro, colunas = previsto):\n")
print(cm_df)

# 10) Exemplo de prediÃ§Ã£o para um novo jogo
novo_jogo = np.array([[1.2, 0.8, 0.3, 0.1]])
novo_jogo_scaled = scaler.transform(novo_jogo)
genero_previsto = le.inverse_transform(knn.predict(novo_jogo_scaled))
print("\nExemplo: gÃªnero previsto para vendas [AN=1.2, EU=0.8, JP=0.3, Outras=0.1] ->", genero_previsto[0])

# ------------------ FIM: BLOCO KNN ------------------
