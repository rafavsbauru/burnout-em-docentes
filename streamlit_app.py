import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import warnings
import re 
from scipy.stats import mannwhitneyu, pearsonr, kruskal # Importações completas

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Dashboard Burnout Docente - Filtros")

# --- INJEÇÃO DE CSS (Fonte 24px) ---
font_size_tabela = "24px" 
st.markdown(f"""
<style>
/* CSS para fonte da tabela */
.stDataFrame table th {{
    font-size: {font_size_tabela} !important;
    font-weight: bold !important; 
    padding: 10px 5px !important; 
}}
.stDataFrame table td {{
    font-size: {font_size_tabela} !important;
    padding: 10px 5px !important;
}}
.stTable table th,
.stTable table td
{{
    font-size: {font_size_tabela} !important;
    padding: 10px 5px !important;
}}
/* CSS para Tabela Vertical Customizada */
.metric-row {{
    display: flex; flex-direction: row; justify-content: space-between; 
    align-items: center; border-bottom: 1px solid #DDDDDD; 
    padding: 12px 5px; 
}}
.metric-label {{
    font-size: 20px; color: #333333; 
}}
.metric-value {{
    font-size: 26px; font-weight: 600; color: #333333; 
}}
.metric-value-n4 {{ color: #FF9800 !important; }}
.metric-value-n5 {{ color: #F44336 !important; }}
</style>
""", unsafe_allow_html=True)
# --- FIM CSS ---

st.title("CARGA DE TRABALHO E SÍNDROME DE BURNOUT EM PROFESSORES")
st.markdown("Análise Exploratória por Segmentação (Filtros)")

# --- Função SIMPLES para carregar dados JÁ LIMPOS do disco ---
@st.cache_data
def load_cleaned_data_from_disk(filepath='cleaned_data.csv'):
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';')
    except FileNotFoundError:
        st.error(f"Erro Crítico: O arquivo '{filepath}' não foi encontrado.")
        st.info("Execute a Célula 2 (Limpeza) primeiro.")
        return None
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return None
    return df

# --- Função para criar Faixas (Bins) para filtros numéricos (CORRIGIDA) ---
def criar_faixas_filtros(df):
    try:
        LABEL_AUSENTE = "Não Informado" 

        # Faixas Numéricas (Idade, Tempo, Carga)
        if 'b1_1_idade' in df.columns:
            bins_idade = [0, 20, 30, 40, 50, 60, 100]
            labels_idade = ['Até 20 anos', '21-30 anos', '31-40 anos', '41-50 anos', '51-60 anos', '61+ anos']
            df['Faixa_Etaria'] = pd.cut(df['b1_1_idade'], bins=bins_idade, labels=labels_idade, right=True).astype('object').fillna(LABEL_AUSENTE)
        
        if 'b3_2_tempo_profissao' in df.columns:
            bins_tempo = [-1, 0, 5, 10, 20, 50]
            labels_tempo = ['0 anos', '1-5 anos', '6-10 anos', '11-20 anos', '21+ anos']
            df['Faixa_Tempo_Profissao'] = pd.cut(df['b3_2_tempo_profissao'], bins=bins_tempo, labels=labels_tempo, right=True).astype('object').fillna(LABEL_AUSENTE)
        
        if 'b3_5_carga_horaria' in df.columns:
            bins_carga = [0, 20, 30, 40, 50, 150]
            labels_carga = ['Até 20h', '21-30h', '31-40h', '41-50h', '51+h']
            df['Faixa_Carga_Horaria'] = pd.cut(df['b3_5_carga_horaria'], bins=bins_carga, labels=labels_carga, right=True).astype('object').fillna(LABEL_AUSENTE)
        
        # Mapear colunas Sim/Não (0/1)
        if 'b4_4_violencia_trabalho' in df.columns:
            df['Filtro_Violencia'] = df['b4_4_violencia_trabalho'].map({0.0: 'Não', 1.0: 'Sim'}).fillna(LABEL_AUSENTE)
        if 'b2_1_acompanhamento_agrupado' in df.columns:
            df['Filtro_Acompanhamento'] = df['b2_1_acompanhamento_agrupado'].map({0.0: 'Não', 1.0: 'Sim'}).fillna(LABEL_AUSENTE)
        if 'b4_3_cultura_feedback' in df.columns:
            df['Filtro_Feedback'] = df['b4_3_cultura_feedback'].map({0.0: 'Não', 1.0: 'Sim'}).fillna(LABEL_AUSENTE)
        
        # Mapear colunas de Instituição (0/1/2)
        if 'b3_7_grupo_instituicao' in df.columns:
            df['Filtro_Instituicao'] = df['b3_7_grupo_instituicao'].map({0.0: 'Somente Pública', 1.0: 'Somente Privada', 2.0: 'Ambas (Pública e Privada)'}).fillna(LABEL_AUSENTE)
        
        # --- INÍCIO DA MUDANÇA (Adicionar filtros de Escala 1-5) ---
        mapa_escala_5pontos_texto = {
            1.0: 'Nunca', 
            2.0: 'Raramente', 
            3.0: 'Às vezes', # Usamos 'Ás vezes' na Célula 2, mas o valor salvo é 3.0
            4.0: 'Frequentemente', 
            5.0: 'Sempre'
        }
        if 'b2_2_frequencia_autocuidado' in df.columns:
            df['Filtro_Autocuidado'] = df['b2_2_frequencia_autocuidado'].map(mapa_escala_5pontos_texto).fillna(LABEL_AUSENTE)
        if 'b2_3_tempo_energia_lazer' in df.columns:
            df['Filtro_Lazer'] = df['b2_3_tempo_energia_lazer'].map(mapa_escala_5pontos_texto).fillna(LABEL_AUSENTE)
        if 'b4_7_apoio_gestao_escolar' in df.columns:
            df['Filtro_Apoio_Gestao'] = df['b4_7_apoio_gestao_escolar'].map(mapa_escala_5pontos_texto).fillna(LABEL_AUSENTE)
        # --- FIM DA MUDANÇA ---
            
    except Exception as e:
        st.error(f"Erro ao criar faixas de filtro: {e}")
        
    return df

# --- Código Principal da Dashboard ---
df = load_cleaned_data_from_disk()

if df is not None and not df.empty:
    st.success("Arquivo 'cleaned_data.csv' carregado com sucesso!")
    df = criar_faixas_filtros(df)
    
    # Atualiza a lista de colunas necessárias
    colunas_necessarias = [
        'b1_2_genero', 'b3_3_nivel_ensino', 'Nivel_Burnout', 'ET',
        'Faixa_Etaria', 'Faixa_Tempo_Profissao', 'Faixa_Carga_Horaria',
        'Filtro_Violencia', 'Filtro_Acompanhamento', 'Filtro_Feedback', 'Filtro_Instituicao',
        'Filtro_Autocuidado', 'Filtro_Lazer', 'Filtro_Apoio_Gestao' # Adicionadas
    ]
    colunas_faltando = [col for col in colunas_necessarias if col not in df.columns]

    if colunas_faltando:
        st.error(f"Erro CSV: Colunas essenciais para os filtros ausentes: {', '.join(colunas_faltando)}")
    else:
        df['Nivel_Burnout'] = df['Nivel_Burnout'].astype(str)
        df['b3_3_nivel_ensino'] = df['b3_3_nivel_ensino'].astype(str).fillna("Não Informado")
        df['b1_2_genero'] = df['b1_2_genero'].astype(str).fillna("Não Informado")

        # --- BARRA LATERAL: Filtros de Segmentação (Versão 1.3 - Expandida) ---
        st.sidebar.header("Filtros de Segmentação")
        try:
            # --- Filtros Demográficos ---
            generos = ['Todos'] + sorted(list(df['b1_2_genero'].dropna().unique()))
            genero_selecionado = st.sidebar.selectbox("Gênero:", generos)
            
            ordem_idade = ['Até 20 anos', '21-30 anos', '31-40 anos', '41-50 anos', '51-60 anos', '61+ anos', 'Não Informado']
            opcoes_idade_existentes = [label for label in ordem_idade if label in df['Faixa_Etaria'].unique()]
            faixas_etarias = ['Todos'] + opcoes_idade_existentes
            faixa_etaria_selecionada = st.sidebar.selectbox("Faixa Etária:", faixas_etarias)

            ordem_tempo = ['0 anos', '1-5 anos', '6-10 anos', '11-20 anos', '21+ anos', 'Não Informado']
            opcoes_tempo_existentes = [label for label in ordem_tempo if label in df['Faixa_Tempo_Profissao'].unique()]
            faixas_tempo = ['Todos'] + opcoes_tempo_existentes
            faixa_tempo_selecionada = st.sidebar.selectbox("Tempo de Profissão:", faixas_tempo)

            # --- Filtros de Atuação (Demandas) ---
            all_options = set()
            for item in df['b3_3_nivel_ensino'].dropna().unique():
                parts = [part.strip() for part in re.split(r'\s*;\s*', str(item))]
                all_options.update(parts)
            unique_niveis = sorted([opt for opt in all_options if opt and opt.lower() != 'nan' and opt.lower() != 'aposentada'])
            niveis_selecionados = st.sidebar.multiselect("Nível(is) de Ensino:", options=unique_niveis, default=[])
            
            ordem_carga = ['Até 20h', '21-30h', '31-40h', '41-50h', '51+h', 'Não Informado']
            opcoes_carga_existentes = [label for label in ordem_carga if label in df['Faixa_Carga_Horaria'].unique()]
            faixas_carga = ['Todos'] + opcoes_carga_existentes
            faixa_carga_selecionada = st.sidebar.selectbox("Faixa Carga Horária:", faixas_carga)

            opcoes_instituicao = ['Todos'] + sorted(list(df['Filtro_Instituicao'].unique()))
            instituicao_selecionada = st.sidebar.selectbox("Tipo de Instituição:", opcoes_instituicao)

            opcoes_violencia = ['Todos'] + sorted(list(df['Filtro_Violencia'].unique()))
            violencia_selecionada = st.sidebar.selectbox("Sofreu Violência?", opcoes_violencia)

            # --- Filtros de Recursos (Pessoais e Org.) ---
            ordem_escala_5p = ['Nunca', 'Raramente', 'Às vezes', 'Frequentemente', 'Sempre', 'Não Informado']
            
            opcoes_autocuidado = ['Todos'] + [label for label in ordem_escala_5p if label in df['Filtro_Autocuidado'].unique()]
            autocuidado_selecionado = st.sidebar.selectbox("Frequência de Autocuidado:", opcoes_autocuidado)

            opcoes_lazer = ['Todos'] + [label for label in ordem_escala_5p if label in df['Filtro_Lazer'].unique()]
            lazer_selecionado = st.sidebar.selectbox("Tempo/Energia para Lazer:", opcoes_lazer)

            opcoes_apoio_gestao = ['Todos'] + [label for label in ordem_escala_5p if label in df['Filtro_Apoio_Gestao'].unique()]
            apoio_gestao_selecionado = st.sidebar.selectbox("Apoio da Gestão?", opcoes_apoio_gestao)

            opcoes_feedback = ['Todos'] + sorted(list(df['Filtro_Feedback'].unique()))
            feedback_selecionado = st.sidebar.selectbox("Cultura de Feedback?", opcoes_feedback)
            
            opcoes_acompanhamento = ['Todos'] + sorted(list(df['Filtro_Acompanhamento'].unique()))
            acompanhamento_selecionado = st.sidebar.selectbox("Faz Acompanhamento?", opcoes_acompanhamento)

        except Exception as e_sidebar:
            st.sidebar.error(f"Erro ao criar filtros: {e_sidebar}")
            # Reseta todos para 'Todos'
            genero_selecionado = 'Todos'; niveis_selecionados = []
            faixa_etaria_selecionada = 'Todas'; faixa_tempo_selecionada = 'Todas'
            faixa_carga_selecionada = 'Todas'; violencia_selecionada = 'Todos'
            acompanhamento_selecionado = 'Todos'; instituicao_selecionada = 'Todos'
            autocuidado_selecionado = 'Todos'; lazer_selecionado = 'Todos'
            apoio_gestao_selecionado = 'Todos'; feedback_selecionado = 'Todos'

        # Aplicação dos Filtros
        df_filtrado = df.copy()
        filtros_aplicados_texto = []
        try:
            if genero_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['b1_2_genero'] == genero_selecionado]
                filtros_aplicados_texto.append(f"Gênero: {genero_selecionado}")
            if niveis_selecionados:
                for nivel in niveis_selecionados:
                    df_filtrado = df_filtrado[df_filtrado['b3_3_nivel_ensino'].str.contains(re.escape(nivel), na=False, case=False, regex=True)]
                filtros_aplicados_texto.append(f"Nível(is): {', '.join(niveis_selecionados)}")
            if faixa_etaria_selecionada != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Faixa_Etaria'] == faixa_etaria_selecionada]
                filtros_aplicados_texto.append(f"Idade: {faixa_etaria_selecionada}")
            if faixa_tempo_selecionada != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Faixa_Tempo_Profissao'] == faixa_tempo_selecionada]
                filtros_aplicados_texto.append(f"Tempo Prof.: {faixa_tempo_selecionada}")
            if faixa_carga_selecionada != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Faixa_Carga_Horaria'] == faixa_carga_selecionada]
                filtros_aplicados_texto.append(f"Carga: {faixa_carga_selecionada}")
            if violencia_selecionada != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Filtro_Violencia'] == violencia_selecionada]
                filtros_aplicados_texto.append(f"Violência: {violencia_selecionada}")
            if acompanhamento_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Filtro_Acompanhamento'] == acompanhamento_selecionado]
                filtros_aplicados_texto.append(f"Acompanhamento: {acompanhamento_selecionado}")
            if instituicao_selecionada != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Filtro_Instituicao'] == instituicao_selecionada]
                filtros_aplicados_texto.append(f"Instituição: {instituicao_selecionada}")
            
            # --- INÍCIO DA MUDANÇA (Aplicando novos filtros) ---
            if autocuidado_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Filtro_Autocuidado'] == autocuidado_selecionado]
                filtros_aplicados_texto.append(f"Autocuidado: {autocuidado_selecionado}")
            if lazer_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Filtro_Lazer'] == lazer_selecionado]
                filtros_aplicados_texto.append(f"Lazer: {lazer_selecionado}")
            if apoio_gestao_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Filtro_Apoio_Gestao'] == apoio_gestao_selecionado]
                filtros_aplicados_texto.append(f"Apoio Gestão: {apoio_gestao_selecionado}")
            if feedback_selecionado != 'Todos':
                df_filtrado = df_filtrado[df_filtrado['Filtro_Feedback'] == feedback_selecionado]
                filtros_aplicados_texto.append(f"Feedback: {feedback_selecionado}")
            # --- FIM DA MUDANÇA ---
                
        except Exception as e_filter:
            st.error(f"Erro ao aplicar filtros: {e_filter}")
            df_filtrado = pd.DataFrame() 

        # Exibir Resultados
        
        # 1. Cabeçalho (N e Filtros Aplicados)
        st.subheader(f"Resultados (N = {len(df_filtrado)})")
        if filtros_aplicados_texto:
            st.info(f"Filtros aplicados: {'; '.join(filtros_aplicados_texto)}")
        else:
            st.info("Mostrando resultados para todos os participantes (nenhum filtro aplicado).")
        
        st.markdown("---") # Linha divisória

        # 2. Visualização (APENAS SE O FILTRO NÃO ZEROU A AMOSTRA)
        if not df_filtrado.empty:
            
            # --- VISUALIZAÇÃO 1: DISTRIBUIÇÃO BURNOUT (VEM PRIMEIRO) ---
            if 'Nivel_Burnout' in df_filtrado.columns:
                df_validos = df_filtrado[~df_filtrado['Nivel_Burnout'].isin(['Erro', 'Inválido', 'nan'])]
                if not df_validos.empty:
                    st.markdown("### Distribuição Burnout")
                    contagem_niveis = df_validos['Nivel_Burnout'].value_counts()
                    count_n1 = contagem_niveis.get('Nível 1', 0)
                    count_n2 = contagem_niveis.get('Nível 2', 0)
                    count_n3 = contagem_niveis.get('Nível 3', 0)
                    count_n4 = contagem_niveis.get('Nível 4', 0)
                    count_n5 = contagem_niveis.get('Nível 5', 0)
                    
                    st.markdown("##### Frequência (Nº de Professores) por Nível:")
                    st.markdown(f"""
                    <div class="metric-row"><span class="metric-label">Nível 1 (Nenhum Indício)</span><span class="metric-value">{count_n1}</span></div>
                    <div class="metric-row"><span class="metric-label">Nível 2 (Possibilidade)</span><span class="metric-value">{count_n2}</span></div>
                    <div class="metric-row"><span class="metric-label">Nível 3 (Fase Inicial)</span><span class="metric-value">{count_n3}</span></div>
                    <div class="metric-row"><span class="metric-label">Nível 4 (Instalação)</span><span class="metric-value metric-value-n4">{count_n4}</span></div>
                    <div class="metric-row"><span class="metric-label">Nível 5 (Fase Considerável)</span><span class="metric-value metric-value-n5">{count_n5}</span></div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True) 

                    # Gráfico de Barras de Distribuição
                    try:
                        fig, ax = plt.subplots(figsize=(10, 5)); cores = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336', '#B71C1C']
                        ordem_niveis = ['Nível 1', 'Nível 2', 'Nível 3', 'Nível 4', 'Nível 5']
                        contagem_ordenada = pd.Series([count_n1, count_n2, count_n3, count_n4, count_n5], index=ordem_niveis)
                        bars = ax.bar(contagem_ordenada.index, contagem_ordenada.values, color=cores)
                        ax.set_title(f"Distribuição (N={len(df_validos)})"); ax.set_xlabel("Nível Risco"); ax.set_ylabel("Contagem")
                        total_validos = len(df_validos)
                        custom_labels = [f"{count} ({((count/total_validos)*100):.1f}%)" if total_validos > 0 else f"{count} (0.0%)" for count in contagem_ordenada]
                        ax.bar_label(bars, labels=custom_labels, label_type='edge', padding=3, fontsize=9); st.pyplot(fig)
                    except Exception as e_plot: st.error(f"Gráfico: {e_plot}")
                    
                    # Métrica
                    try:
                        risco45 = count_n4 + count_n5
                        total_validos_para_perc = total_validos if total_validos > 0 else 1
                        perc45 = (risco45 / total_validos_para_perc) * 100 if total_validos > 0 else 0
                        st.metric("Risco Alto/Crítico (Níveis 4/5)", f"{perc45:.1f}%", f"Total: {risco45}", delta_color="inverse")
                    except Exception as e_metric: st.error(f"Métrica: {e_metric}")
                else: st.warning("Sem dados válidos de Nível Burnout no grupo filtrado.")
            else: st.error("Coluna 'Nivel_Burnout' não encontrada.")
            
            st.markdown("---") # Linha divisória

            # --- VISUALIZAÇÃO 2: TESTE DE SIGNIFICÂNCIA DO FILTRO (VEM DEPOIS) ---
            if filtros_aplicados_texto: # Só roda se houver filtro
                indices_filtrados = df_filtrado.index
                df_restante = df.drop(indices_filtrados) 
                
                if not df_restante.empty and not df_filtrado.empty and 'ET' in df_filtrado.columns and 'ET' in df_restante.columns:
                    mediana_grupo = df_filtrado['ET'].median()
                    mediana_restante = df_restante['ET'].median()
                    
                    try:
                                # 1. Cabeçalho (N e Filtros Aplicados)
                        st.subheader(f"Resultados (N = {len(df_filtrado)})")
                        if filtros_aplicados_texto:
                            st.info(f"Filtros aplicados: {'; '.join(filtros_aplicados_texto)}")
                        else:
                            st.info("Mostrando resultados para todos os participantes (nenhum filtro aplicado).")
        
        
                        stat, p_value = mannwhitneyu(df_filtrado['ET'].dropna(), df_restante['ET'].dropna(), alternative='two-sided')
                        
                        st.markdown("##### Análise do Grupo Filtrado:")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        if p_value < 0.001:
                            p_text_metric = "< 0.001"
                        else:
                            p_text_metric = f"{p_value:.4f}"
                            
                        col1.metric(label="Mediana (Grupo Filtrado)", value=f"{mediana_grupo:.2f}")
                        col2.metric(label="Mediana (Restante da Amostra)", value=f"{mediana_restante:.2f}")
                        col3.metric(label="P-valor (Mann-Whitney U)", value=p_text_metric)

                        if p_value < 0.05:
                            p_text_display = f"(p {p_text_metric})"
                            st.success(f"**Achado Significativo {p_text_display}.** Os resultados observados a partir dos dados do grupo filtrado (Mediana = {mediana_grupo:.2f}) e do restante da amostra (Mediana = {mediana_restante:.2f}), pelo Teste de Mann-Whitney U, **apresentaram-se estatisticamente significativos**.")
                        
                        else: # Se Não Significativo
                            p_text_display = f"(p = {p_text_metric})"
                            st.warning(f"**Achado Não Significativo {p_text_display}.** Os resultados observados a partir dos dados do grupo filtrado (Mediana = {mediana_grupo:.2f}) e do restante da amostra (Mediana = {mediana_restante:.2f}), pelo Teste de Mann-Whitney U, **não se apresentaram estatisticamente significativos**.")
                    
                    except ValueError:
                        st.warning("Não é possível comparar (um dos grupos pode não ter variação).")
                else:
                    st.warning("Não é possível comparar (grupo filtrado ou grupo restante está vazio ou sem dados de ET).")
            
        else: # Mensagem se o filtro zerou a amostra
            st.info("Nenhum professor corresponde aos filtros selecionados.")
else:
    st.info("👈 Carregue o arquivo CSV LIMPO ('cleaned_data.csv') na barra lateral para iniciar.")
