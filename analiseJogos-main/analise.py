#importando as bibliotecas
import os
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np
import plotly.express as px
import plotly.io as pio
from matplotlib.backends.backend_pdf import PdfPages
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


#GRAFICOS de barras horizontal com os dez jogos mais vendidos EM MILHOES
top_10 = df.sort_values('vendas totais', ascending=False).head(10)
plt.figure(figsize=(10,7))
sns.barplot(x='vendas totais', y='jogo', data=top_10)
plt.ylabel('Jogos', size=13)
plt.xlabel('Total de vendas (em milhões)', size=13)
plt.title('Os dez jogos mais vendidos entre 1980 e 2016', size=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()

#GRAFICO: Consoles com mais lançamentos (top 10)
most_releases = df['console'].value_counts().head(10).reset_index()
most_releases.columns = ['console', 'lançamentos']
plt.figure(figsize=(10,7))
sns.barplot(x='lançamentos', y='console', data=most_releases)
plt.xlabel('Número de lançamentos', size=13)
plt.ylabel('Consoles', size=13)
plt.title('Consoles com maior número de lançamentos', size=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#GRAFICOS Consoles que mais venderam jogos EM MILHOES
most_sales = pd.DataFrame(df.groupby('console')[['vendas totais']].sum().sort_values('vendas totais', ascending = False).head(10))
plt.figure(figsize=(10,7))
sns.barplot(x = most_sales.index, y = 'vendas totais', data = most_sales)
plt.ylabel('Jogos vendidos (em milhões)', size = 13)
plt.xlabel('consoles', size = 13)
plt.title('Consoles que mais venderam jogos', size = 15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, rotation=80)
plt.show()

#GRAFICO frequencia anual de lancamento
plt.figure(figsize=(10,7))
sns.histplot(data=df, x='ano de lancamento', bins=35, kde=True)
plt.ylabel('Lançamentos', size=13)
plt.xlabel('Ano', size=13)
plt.title('Frequência anual de lançamentos', size=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, rotation=80)
plt.show()

#GRAFICO DE VENDAS POR GENERO PIRÂMIDE
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

# ------------------ INÍCIO: BLOCO KNN ------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# 1) Features e alvo
X = df[['vendas américa do norte', 'vendas europa', 'vendas japao', 'vendas outras regioes']].copy()
y = df['genero'].copy()

# 2) Limpeza
mask_valid = y.notna() & (X.sum(axis=1) > 0)
X = X[mask_valid]
y = y[mask_valid]

# 3) Codificação dos gêneros
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

# 7) Predição e avaliação
y_pred = knn.predict(X_test_scaled)

print("\n=== RESULTADOS KNN (vendas por região -> prever gênero) ===")
print("Número de amostras (treino):", X_train.shape[0], "| (teste):", X_test.shape[0])
print("Acurácia no conjunto de teste: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# 8) Relatório em português
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

print("\nRelatório de classificação (em português):\n")
print(df_relatorio.round(2))

# 9) Matriz de confusão com nomes dos gêneros
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

print("\nMatriz de Confusão (linhas = verdadeiro, colunas = previsto):\n")
print(cm_df)

# 10) Exemplo de predição para um novo jogo
novo_jogo = pd.DataFrame([[1.2, 0.8, 0.3, 0.1]], columns=X.columns)
novo_jogo_scaled = scaler.transform(novo_jogo)
genero_previsto = le.inverse_transform(knn.predict(novo_jogo_scaled))
print("\nExemplo: gênero previsto para vendas [AN=1.2, EU=0.8, JP=0.3, Outras=0.1] ->", genero_previsto[0])

# ------------------ FIM: BLOCO KNN ------------------

# ------------------ INICIO: BLOCO REGRESSAO ------------------
# Objetivo: prever vendas totais (em milhoes) usando informacoes de ano, console, genero e publicadora.
# Modelo escolhido: RandomForestRegressor para lidar bem com relacionamentos nao lineares e categorias via one-hot.

# 1) Seleciona features e alvo
features_reg = ['ano de lancamento', 'console', 'genero', 'publicadora']
target_reg = 'vendas totais'
df_reg = df[features_reg + [target_reg]].copy()
df_reg = df_reg.dropna(subset=[target_reg])
df_reg = df_reg[df_reg[target_reg] > 0]

X_reg = df_reg[features_reg]
y_reg = df_reg[target_reg]

# 2) Treino e teste
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# 3) Pipeline de preparo (one-hot para categorias, passthrough para numerico)
colunas_categoricas = ['console', 'genero', 'publicadora']
colunas_numericas = ['ano de lancamento']

preprocessador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas),
        ('num', 'passthrough', colunas_numericas)
    ]
)

modelo_reg = RandomForestRegressor(
    n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2
)

pipeline_reg = Pipeline(steps=[
    ('preprocessamento', preprocessador),
    ('modelo', modelo_reg)
])

# 4) Treina e avalia
pipeline_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = pipeline_reg.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n=== RESULTADOS REGRESSAO (prever vendas totais) ===")
print("Amostras -> treino:", X_train_reg.shape[0], "| teste:", X_test_reg.shape[0])
print("MAE: {:.3f} milhoes".format(mae))
print("RMSE: {:.3f} milhoes".format(rmse))
print("R2: {:.3f}".format(r2))

# 5) Exemplo de previsao para novo jogo
exemplo_reg = pd.DataFrame([{
    'ano de lancamento': 2016,
    'console': 'PS4',
    'genero': 'Action',
    'publicadora': 'Sony Computer Entertainment'
}])
prev_exemplo = pipeline_reg.predict(exemplo_reg)[0]
print("\nExemplo: vendas previstas para jogo (PS4, Action, 2016, Sony) -> {:.3f} milhoes".format(prev_exemplo))

# ------------------ FIM: BLOCO REGRESSAO ------------------

# ------------------ INICIO: EXPORTACAO DE GRAFICOS PARA PDF ------------------
# Gera um PDF em ./relatorio/relatorio.pdf com alguns graficos de visao geral.

os.makedirs("relatorio", exist_ok=True)
pdf_caminho = os.path.join("relatorio", "relatorio.pdf")

with PdfPages(pdf_caminho) as pdf:
    # Grafico: top 10 jogos mais vendidos
    top_10 = df.sort_values('vendas totais', ascending=False).head(10)
    plt.figure(figsize=(10, 7))
    sns.barplot(x='vendas totais', y='jogo', data=top_10, color="steelblue")
    plt.ylabel('Jogos', size=12)
    plt.xlabel('Total de vendas (milhoes)', size=12)
    plt.title('Top 10 jogos mais vendidos (1980-2016)', size=14)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Grafico: frequencia anual de lancamento
    plt.figure(figsize=(10, 7))
    sns.histplot(data=df, x='ano de lancamento', bins=35, kde=True, color="darkorange")
    plt.ylabel('Lancamentos', size=12)
    plt.xlabel('Ano', size=12)
    plt.title('Frequencia anual de lancamentos', size=14)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"\nPDF de graficos gerado em: {pdf_caminho}")
# ------------------ FIM: EXPORTACAO DE GRAFICOS PARA PDF ------------------
