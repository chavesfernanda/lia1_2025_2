import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
from io import BytesIO
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableRow, TableCell
from odf.text import P


# Configuração da página

st.set_page_config(page_title="Simulador de Compra de Veículo", layout="wide")
st.title("🚗 Simulador Inteligente de Compra de Veículo")

st.markdown("""
Este simulador avalia a **qualidade**, estima o **custo total** e exibe **modelos reais compatíveis**
de acordo com as características desejadas, gerando relatórios analíticos e histórico automático em
**LibreOffice (.ods)** e **CSV**.
""")

# 1️⃣ Modelos de Machine Learning

@st.cache_data
def load_classification_model():
    cars = pd.read_csv("avaliacao_veiculo.csv", sep=";")
    encoder = OrdinalEncoder()
    for col in cars.columns.drop('evaluation'):
        cars[col] = cars[col].astype('category')
    X_encoded = encoder.fit_transform(cars.drop('evaluation', axis=1))
    y = cars['evaluation'].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    model = CategoricalNB()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return encoder, model, accuracy, cars

encoder, clf_model, clf_accuracy, cars = load_classification_model()

@st.cache_data
def load_regression_model():
    data = pd.read_csv("franquia_custos_iniciais.csv", sep=";")
    X = data[['custo_franquia_anual']]
    y = data['investimento_inicial']
    model = LinearRegression().fit(X, y)
    return model, data

reg_model, data_franquia = load_regression_model()

# 2️⃣ Dataset de modelos reais

@st.cache_data
def load_car_models():
    return pd.read_csv("modelos_carros.csv", sep=";")

try:
    modelos = load_car_models()
except FileNotFoundError:
    st.error("❌ O arquivo 'modelos_carros.csv' não foi encontrado.")
    st.stop()


# 3️⃣ Entradas do cliente

st.header("🧭 Informe as características desejadas")

col1, col2, col3 = st.columns(3)

with col1:
    preco = st.selectbox("Faixa de preço:", cars['buying'].unique())
    manutencao = st.selectbox("Custo de manutenção:", cars['maint'].unique())
    portas = st.selectbox("Número de portas:", cars['doors'].unique())
    cor = st.selectbox("Cor desejada:", ["preto", "branco", "prata", "vermelho", "azul"])

with col2:
    assentos = st.selectbox("Número de assentos:", cars['seats'].unique())
    porta_malas = st.selectbox("Tamanho do porta-malas:", cars['lug_boot'].unique())
    seguranca = st.selectbox("Nível de segurança:", cars['safety'].unique())
    marca_preferida = st.selectbox("Marca preferida:", ["Fiat", "Toyota", "Honda", "Chevrolet", "Hyundai", "Jeep", "Renault", "Volkswagen"])

with col3:
    combustivel = st.selectbox("Tipo de combustível:", ["gasolina", "etanol", "diesel", "elétrico", "híbrido"])
    cambio = st.selectbox("Tipo de câmbio:", ["manual", "automático", "CVT"])
    ano_modelo = st.slider("Ano do modelo:", 2010, 2025, 2023)
    consumo_medio = st.slider("Consumo médio (km/l):", 5, 25, 12)

valor_franquia = st.number_input(
    "Custo anual estimado (R$):", min_value=1000.0, max_value=100000.0, value=3000.0, step=100.0
)


# 4️⃣ Processamento e resultados

if st.button("Buscar Veículos"):
    input_df = pd.DataFrame([[preco, manutencao, portas, assentos, porta_malas, seguranca]],
                            columns=cars.columns.drop("evaluation"))
    input_encoded = encoder.transform(input_df)
    pred_encoded = clf_model.predict(input_encoded)
    qualidade = cars['evaluation'].astype('category').cat.categories[pred_encoded][0]
    custo_previsto = reg_model.predict([[valor_franquia]])[0]

    st.subheader("🔍 Resultado da Avaliação")
    st.write(f"**Qualidade estimada:** {qualidade}")
    st.write(f"**Custo total estimado:** R$ {custo_previsto:,.2f}")
    st.caption(f"Acurácia do modelo de classificação: {clf_accuracy:.2f}")

    # --- Exibir modelos compatíveis ---
    st.markdown("---")
    st.subheader("🚘 Modelos compatíveis com sua preferência:")

    filtro = (
        (modelos["preco"].str.lower() == preco.lower()) &
        (modelos["manutencao"].str.lower() == manutencao.lower()) &
        (modelos["portas"].astype(str) == str(portas)) &
        (modelos["assentos"].astype(str) == str(assentos)) &
        (modelos["porta_malas"].str.lower() == porta_malas.lower()) &
        (modelos["seguranca"].str.lower() == seguranca.lower()) &
        (modelos["combustivel"].str.lower() == combustivel.lower()) &
        (modelos["cambio"].str.lower() == cambio.lower()) &
        (modelos["cor"].str.lower() == cor.lower()) &
        (modelos["marca"].str.lower() == marca_preferida.lower()) &
        (modelos["consumo_medio"].astype(float) >= consumo_medio - 2) &
        (modelos["ano_modelo"].astype(int) >= int(ano_modelo) - 1)
    )
    carros_filtrados = modelos[filtro]

# --- Carregar CSV ---
df = pd.read_csv("modelos_carros.csv", sep=';')

# Padronizando colunas de texto para lowercase para facilitar filtros
for col in ['preco', 'manutencao', 'porta_malas', 'seguranca', 'combustivel', 'cambio', 'cor', 'marca', 'modelo']:
    df[col] = df[col].astype(str).str.lower()

# --- Filtros do usuário ---
preco_selecionado = st.selectbox("Preço", ["low", "med", "high", "vhigh"])
combustivel_selecionado = st.selectbox("Combustível", ["gasolina", "etanol", "diesel", "híbrido"])
ano_min = st.number_input("Ano mínimo", min_value=2000, max_value=2025, value=2020)

# Aplicando filtro
carros_filtrados = df[
    (df['preco'] == preco_selecionado.lower()) &
    (df['combustivel'] == combustivel_selecionado.lower()) &
    (df['ano_modelo'] >= ano_min)
]

# Debug rápido
st.write(f"Carros encontrados: {len(carros_filtrados)}")

# --- Exibição dos carros ---
if len(carros_filtrados) > 0:
    for _, carro in carros_filtrados.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            if os.path.exists(carro["imagem"]):
                st.image(carro["imagem"], width=180)
            else:
                st.image("https://via.placeholder.com/180x120?text=Sem+Imagem", width=180)
        with col2:
            st.markdown(f"### {carro['modelo'].capitalize()} ({carro['marca'].capitalize()})")
            st.write(f"💰 **Preço:** {carro['preco'].capitalize()} | 🛠 **Manutenção:** {carro['manutencao'].capitalize()}")
            st.write(f"🧍 **Assentos:** {carro['assentos']} | 🚪 Portas: {carro['portas']}")
            st.write(f"🧳 **Porta-malas:** {carro['porta_malas'].capitalize()} | 🛡 Segurança: {carro['seguranca'].capitalize()}")
            st.write(f"⛽ **Combustível:** {carro['combustivel'].capitalize()} | ⚙️ **Câmbio:** {carro['cambio'].capitalize()}")
            st.write(f"📅 **Ano:** {carro['ano_modelo']} | 🎨 **Cor:** {carro['cor'].capitalize()} | 🏁 **Consumo:** {carro['consumo_medio']} km/l")
            st.markdown("---")
else:
    st.warning("Nenhum modelo encontrado com essas características.")


    # --- Registro da simulação ---
    registro = {
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preco": preco,
        "manutencao": manutencao,
        "portas": portas,
        "assentos": assentos,
        "porta_malas": porta_malas,
        "seguranca": seguranca,
        "combustivel": combustivel,
        "cambio": cambio,
        "ano_modelo": ano_modelo,
        "consumo_medio": consumo_medio,
        "cor": cor,
        "marca_preferida": marca_preferida,
        "custo_anual": valor_franquia,
        "qualidade": qualidade,
        "custo_estimado": round(custo_previsto, 2)
    }
    historico_file = "historico_simulacoes.csv"
    if os.path.exists(historico_file):
        df_hist = pd.read_csv(historico_file)
        df_hist = pd.concat([df_hist, pd.DataFrame([registro])], ignore_index=True)
    else:
        df_hist = pd.DataFrame([registro])
    df_hist.to_csv(historico_file, index=False)

    # --- Salvar automaticamente em ODS ---
    ods_file_path = "historico_simulacoes.ods"
    ods_doc = OpenDocumentSpreadsheet()
    table = Table(name="Histórico")

    header_row = TableRow()
    for col in df_hist.columns:
        cell = TableCell()
        cell.addElement(P(text=col))
        header_row.addElement(cell)
    table.addElement(header_row)

    for _, row in df_hist.iterrows():
        tr = TableRow()
        for value in row:
            cell = TableCell()
            cell.addElement(P(text=str(value)))
            tr.addElement(cell)
        table.addElement(tr)

    ods_doc.spreadsheet.addElement(table)
    ods_doc.save(ods_file_path)
    st.success("✅ Histórico atualizado e salvo automaticamente em formato LibreOffice (.ods)!")

# 5️⃣ Painel de histórico e gráficos

st.markdown("---")
st.header("📊 Resumo das Simulações e Análises")

if os.path.exists("historico_simulacoes.csv"):
    df_hist = pd.read_csv("historico_simulacoes.csv")

    st.dataframe(df_hist.sort_values("data", ascending=False), use_container_width=True)

    st.subheader("📈 Quantidade de Simulações por Qualidade")
    fig1, ax1 = plt.subplots()
    df_hist["qualidade"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Qualidade do veículo")
    ax1.set_ylabel("Número de simulações")
    st.pyplot(fig1)

    st.subheader("🧁 Distribuição Percentual das Qualidades")
    fig_pie, ax_pie = plt.subplots()
    df_hist["qualidade"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax_pie)
    ax_pie.set_ylabel("")
    st.pyplot(fig_pie)

    st.subheader("💰 Custo médio estimado por qualidade")
    fig2, ax2 = plt.subplots()
    df_hist.groupby("qualidade")["custo_estimado"].mean().plot(kind="bar", color="green", ax=ax2)
    ax2.set_ylabel("Custo médio (R$)")
    st.pyplot(fig2)

    st.subheader("🕒 Evolução das Simulações ao longo do tempo")
    df_hist["data"] = pd.to_datetime(df_hist["data"])
    df_hist["dia"] = df_hist["data"].dt.date
    fig3, ax3 = plt.subplots()
    df_hist.groupby("dia").size().plot(kind="line", marker="o", ax=ax3)
    ax3.set_xlabel("Data")
    ax3.set_ylabel("Número de simulações")
    st.pyplot(fig3)
else:
    st.info("Nenhuma simulação registrada ainda.")