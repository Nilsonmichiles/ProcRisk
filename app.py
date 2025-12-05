
import sys
import warnings
if not sys.modules.get('warnings'):
    sys.modules['warnings'] = warnings
    
import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Painel Análitico de Suporte à Priorização de Auditoria de Contratos")

# --- Estilo Customizado (CSS) ---
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-mod { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# --- Função de Carregamento ---
@st.cache_data
def load_data():
    data = joblib.load('dados_dashboard_fraude.pkl')
    df = data['dataframe']
    
    # --- APLICAÇÃO DAS REGRAS DE DECISÃO (Conforme Texto Acadêmico) ---
    def classificar_risco(score):
        if score >= 0.75:
            return "ALTO"
        elif score >= 0.40:
            return "MODERADO"
        else:
            return "BAIXO"

    df['Nivel_Risco'] = df['Risco_Calculado'].apply(classificar_risco)
    return df, data['shap_values'], data['feature_names']

try:
    df, shap_values_matrix, feature_names = load_data()
except FileNotFoundError:
    st.error("Arquivo 'dados_dashboard_fraude.pkl' não encontrado. Execute o script de exportação primeiro.")
    st.stop()

# --- Sidebar: Filtros de Governança ---
st.sidebar.header("🔍 Filtros de Auditoria")

# 1. Filtro por Nível de Risco (Priorização)
risk_filter = st.sidebar.multiselect(
    "Nível de Risco (Prioridade)",
    options=["ALTO", "MODERADO", "BAIXO"],
    default=["ALTO", "MODERADO"], # Foco inicial nos casos críticos
    help="Alto: Score >= 0.75 | Moderado: 0.40 a 0.75 | Baixo: < 0.40"
)

# 2. Filtro por Score Numérico (Refinamento)
min_score, max_score = st.sidebar.slider(
    "Refinar Score de Risco (%)", 0, 100, (0, 100)
)

# 3. Filtro por Status Real (Ground Truth - para validação)
status_filter = st.sidebar.multiselect(
    "Status Real (Validação)", 
    options=[0, 1], 
    format_func=lambda x: "Fraude Confirmada (1)" if x == 1 else "Regular (0)",
    default=[0, 1]
)

# --- Aplicação dos Filtros ---
df_filtered = df[
    (df['Nivel_Risco'].isin(risk_filter)) &
    (df['Risco_Calculado'] * 100 >= min_score) & 
    (df['Risco_Calculado'] * 100 <= max_score) &
    (df['Fraude_Real'].isin(status_filter))
]

# --- Layout Principal ---
st.title("🛡️ Painel de Governança e Priorização de Auditorias")
st.markdown("Ferramenta de apoio à decisão baseada em risco (Hybrid Ensemble).")

# KPIs do Topo
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Contratos Selecionados", len(df_filtered))
kpi2.metric("Prioridade Alta", len(df_filtered[df_filtered['Nivel_Risco']=="ALTO"]))
kpi3.metric("Média de Risco", f"{df_filtered['Risco_Calculado'].mean():.1%}")
kpi4.download_button(
    label="📥 Exportar Contratos de Selecionados",
    data=df_filtered.to_csv(index=True).encode('utf-8'),
    file_name='ordem_servico_auditoria.csv',
    mime='text/csv',
    help="Gera lista CSV para distribuição aos auditores."
)

# Colunas para layout (Tabela vs Detalhes)
col_table, col_details = st.columns([1.6, 1])

with col_table:
    st.subheader("📋 Fila de Auditoria")
    
    # Configuração de cores para a tabela
    def highlight_risk(val):
        color = 'red' if val == 'ALTO' else 'orange' if val == 'MODERADO' else 'green'
        return f'color: {color}; font-weight: bold'

    # Exibir tabela interativa
    event = st.dataframe(
        df_filtered[['Nivel_Risco', 'Risco_Calculado', 'Fraude_Real'] + feature_names[:5]] # Mostra risco + 5 primeiras features
        .style
        .format({'Risco_Calculado': '{:.2%}'})
        .map(highlight_risk, subset=['Nivel_Risco'])
        .background_gradient(subset=['Risco_Calculado'], cmap='Reds', vmin=0, vmax=1),
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        height=600
    )

# Lógica de Seleção
selected_index = None
if len(event.selection['rows']) > 0:
    row_idx_filtered = event.selection['rows'][0]
    selected_index = df_filtered.index[row_idx_filtered]
else:
    if not df_filtered.empty:
        selected_index = df_filtered.index[0]

# --- Painel de Detalhes (Direita) ---
with col_details:
    if selected_index is not None:
        row_data = df.loc[selected_index]
        
        # Recuperar SHAP
        try:
            pos_idx = df.index.get_loc(selected_index)
            shap_values_single = shap_values_matrix[pos_idx]
        except KeyError:
            st.error("Erro de índice.")
            st.stop()

        # --- 1. Velocímetro (Gauge) com as Regras do Texto ---
        st.subheader("📊 Diagnóstico de Risco")
        
        score = row_data['Risco_Calculado']
        risk_level = row_data['Nivel_Risco']
        
        # Cores baseadas no nível
        gauge_color = "red" if risk_level == "ALTO" else "orange" if risk_level == "MODERADO" else "green"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score * 100,
            title = {'text': f"Nível: {risk_level}"},
            delta = {'reference': 40, 'increasing': {'color': "red"}}, # Referência de corte base
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 40], 'color': "#e6f9e6"},   # Baixo (Verde Claro)
                    {'range': [40, 75], 'color': "#fff2cc"},  # Moderado (Amarelo Claro)
                    {'range': [75, 100], 'color': "#ffcccc"}  # Alto (Vermelho Claro)
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score * 100
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Dados Cadastrais
        st.info(f"🆔 **ID do Contrato/Fornecedor:** {selected_index}")

        # --- 2. Explicabilidade (SHAP Waterfall) ---
        st.subheader("🔍 Fatores Determinantes (SHAP)")
        st.markdown(f"Por que o modelo atribuiu **{score:.1%}** de risco?")

        # Criar objeto Explanation
        shap_exp = shap.Explanation(
            values=shap_values_single,
            base_values=0.5, 
            data=row_data[feature_names].values,
            feature_names=feature_names
        )

        # Plotar
        fig, ax = plt.subplots(figsize=(10, 12))
        shap.plots.waterfall(shap_exp, max_display=8, show=False)
        st.pyplot(fig, bbox_inches='tight')
        
    else:

        st.warning("Nenhum contrato selecionado ou lista vazia.")


