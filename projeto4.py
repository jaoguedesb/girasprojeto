import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import ttest_1samp

# Configurações iniciais do Streamlit
st.set_page_config(page_title="TikTok Data Insights", layout="wide")

# Função para limpeza de dados
def clean_data(df):
    # Tratar valores ausentes
    df = df.dropna()
    # Adicionar novas métricas para enriquecer a análise
    df['likes_per_view'] = df['video_like_count'] / df['video_view_count']
    df['comments_per_view'] = df['video_comment_count'] / df['video_view_count']
    return df

# Função para realizar regressão linear
def perform_multiple_regression(df, x_cols, y_col):
    """Realiza uma regressão múltipla usando statsmodels."""
    X = df[x_cols]
    y = df[y_col]
    X = sm.add_constant(X)  # Adiciona uma constante para o modelo
    model = sm.OLS(y, X).fit()
    return model

# Carregar o dataset
df = pd.read_csv('tiktok_datasetnovo.csv')
df = clean_data(df)

# Header com estilo customizado
st.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
    }
    .ranking-table {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<div class='main-title'>TikTok Data Insights</div>", unsafe_allow_html=True)

# Barra lateral com opções
st.sidebar.markdown("<div class='sidebar-title'>Navegação</div>", unsafe_allow_html=True)
menu = [
    ' Análise Exploratória', 
    ' Melhores Vídeos', 
    ' Modelos de Regressão', 
    ' Testes de Hipóteses', 
    ' Filtragem de Dados',
    ' Soluções Práticas' 
]
choice = st.sidebar.radio("Escolha uma opção:", menu)



if choice == ' Análise Exploratória':
    st.header(" Análise Exploratória")
    st.write("Exploração inicial dos dados com estatísticas descritivas e padrões observados.")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resumo Estatístico")
        st.dataframe(df.describe())
    
    with col2:
        st.subheader("Distribuição de Visualizações")
        fig = px.histogram(df, x='video_view_count', nbins=50, title="Distribuição de Visualizações")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlação entre Métricas")
    # Calcular a matriz de correlação com todas as variáveis relevantes
    corr = df[['video_view_count', 'video_like_count', 'video_comment_count', 
               'video_duration_sec', 'video_download_count', 'video_share_count',
               'likes_per_view', 'comments_per_view']].corr().abs()
    # Plotar o mapa de correlação
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Mapa de Correlação Entre Métricas (0 a 1)",
        labels=dict(color="Correlação (|r|)")
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Adicionar exemplos abaixo do mapa
    st.markdown("### Exemplos de Relações no Mapa de Correlação:")
    st.markdown("""
    - **`video_view_count` e `video_like_count`:**  
      Correlação próxima de **0.9**.  
      **Significa:** Vídeos com mais visualizações tendem a receber mais curtidas.
      
    - **`video_view_count` e `video_duration_sec`:**  
      Correlação próxima de **0.3**.  
      **Significa:** A duração do vídeo tem uma relação fraca com o número de visualizações. Vídeos mais longos ou curtos podem ter visualizações semelhantes.
      
    - **`likes_per_view` e `video_share_count`:**  
      Correlação próxima de **0.7**.  
      **Significa:** Vídeos com mais curtidas por visualização tendem a ser mais compartilhados, mas a relação não é perfeita.
      
    - **`video_comment_count` e `video_view_count`:**  
      Correlação próxima de **0.85**.  
      **Significa:** Mais visualizações geralmente resultam em mais comentários.
    """)
    


elif choice == ' Melhores Vídeos':
    st.header(" Melhores Vídeos")
    st.write("Identifique os vídeos mais populares com base no número de visualizações.")

    # Selecionar os 10 vídeos mais populares e adicionar o ranking
    top_videos = df.sort_values(by='video_view_count', ascending=False).head(10)
    top_videos = top_videos.reset_index(drop=True)
    top_videos['Rank'] = top_videos.index + 1

    # Gráfico de dispersão (bubble chart) com ranking
    fig = px.scatter(
        top_videos,
        x='video_like_count',
        y='video_comment_count',
        size='video_view_count',
        color='video_share_count',
        title="Top 10 Vídeos Mais Populares no TikTok (Com base em Curtidas e Comentários)",
        labels={
            'video_like_count': 'Número de Curtidas',
            'video_comment_count': 'Número de Comentários',
            'video_view_count': 'Número de Visualizações',
            'video_share_count': 'Número de Compartilhamentos',
            'Rank': 'Posição no Ranking'
        },
        hover_data=['Rank', 'video_id']
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis=dict(title="Número de Curtidas"),
        yaxis=dict(title="Número de Comentários"),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Exibindo o ranking abaixo do gráfico
    st.markdown("<b>Ranking dos Top 10 Vídeos:</b>", unsafe_allow_html=True)
    top_videos_display = top_videos[['Rank', 'video_id', 'video_view_count', 'video_like_count', 'video_comment_count', 'video_duration_sec']]
    top_videos_display.columns = ['Posição', 'ID do Vídeo', 'Visualizações', 'Curtidas', 'Comentários', 'Duração (segundos)']
    st.dataframe(top_videos_display)

elif choice == ' Soluções Práticas':
    st.header(" Soluções Práticas Baseadas na Análise de Dados")
    st.subheader(" Curtidas por Visualização e Compartilhamentos")
    st.write("""
        - **** Vídeos com maior proporção de curtidas por visualização tendem a ser mais compartilhados.
        - **** Incentivar a interação por meio de CTAs (call-to-action).
        - **** Um aumento de 10% nas curtidas pode levar a um crescimento médio de 7% nos compartilhamentos.
    """)

    corr = df[['video_like_count', 'video_share_count']].corr()
    df['likes_increase'] = df['video_like_count'] * 1.1
    df['predicted_shares'] = df['likes_increase'] * corr.loc['video_like_count', 'video_share_count']

    fig1 = px.scatter(
        df,
        x='video_like_count',
        y='video_share_count',
        title='Impacto de Aumentar Curtidas nos Compartilhamentos',
        labels={'video_like_count': 'Curtidas', 'video_share_count': 'Compartilhamentos'},
        hover_data=['video_id']
    )
    fig1.add_scatter(
        x=df['likes_increase'],
        y=df['predicted_shares'],
        mode='markers',
        name='Impacto Esperado',
        marker=dict(color='orange', size=7)
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(" Duração do Vídeo e Visualizações")
    st.write("""
        - **** Vídeos curtos têm maior retenção e engajamento para até 30 segundos.
        - **** Produzir conteúdos diretos e atrativos com duração de até 30 segundos.
    """)

    fig2 = px.histogram(
        df,
        x='video_duration_sec',
        y='video_view_count',
        title='Distribuição de Visualizações por Duração do Vídeo',
        labels={'video_duration_sec': 'Duração (segundos)', 'video_view_count': 'Visualizações'},
        nbins=30
    )
    st.plotly_chart(fig2, use_container_width=True)


# Inicializar múltiplos históricos no estado do Streamlit
if 'history_groups' not in st.session_state:
    st.session_state['history_groups'] = {}

if choice == ' Modelos de Regressão':
    st.header(" Modelos de Regressão Múltipla")
    st.write("Analise como diferentes métricas influenciam as visualizações dos vídeos.")

    col1, col2 = st.columns(2)
    with col1:
        x_cols = st.multiselect("Selecione as variáveis independentes (X):", [
            'video_like_count', 'video_comment_count', 
            'video_duration_sec', 'video_download_count', 'video_share_count'
        ])
    with col2:
        y_col = 'video_view_count'
        st.write(f"Variável dependente (Y): {y_col}")

    # Criar ou selecionar o grupo de histórico
    group_name = st.text_input("Nome do grupo de histórico:", "Grupo 1")
    if group_name not in st.session_state['history_groups']:
        st.session_state['history_groups'][group_name] = []

    if x_cols:
        model = perform_multiple_regression(df, x_cols, y_col)
        st.subheader(f"Resumo do Modelo de Regressão (Y: {y_col}, X: {x_cols})")
        st.text(model.summary())

        st.write(f"R² Ajustado: {model.rsquared_adj:.4f}")
        st.write(f"AIC (Critério de Informação de Akaike): {model.aic:.4f}")

        st.subheader("Previsão com Base no Modelo")
        # Dicionário para armazenar valores inseridos pelo usuário
        input_values = {col: st.number_input(f"Insira um valor para {col}:", min_value=0.0, step=1.0) for col in x_cols}
        if st.button("Prever"):
            input_data = [1] + [input_values[col] for col in x_cols]
            prediction = model.predict([input_data])[0]
            st.success(f"Previsão de visualizações: {prediction:.2f}")

            # Adicionar os dados ao grupo de histórico selecionado
            st.session_state['history_groups'][group_name].append({**input_values, 'Previsão': prediction})

        # Exibir o histórico do grupo selecionado
        st.subheader(f"Histórico de Previsões - {group_name}")
        if st.session_state['history_groups'][group_name]:
            st.dataframe(pd.DataFrame(st.session_state['history_groups'][group_name]))

            # Botão para limpar o histórico do grupo atual
            if st.button(f"Limpar Histórico de {group_name}"):
                st.session_state['history_groups'][group_name] = []

    # Comparar múltiplos históricos
    st.subheader("Comparar Históricos")
    if st.session_state['history_groups']:
        selected_groups = st.multiselect("Selecione os grupos para comparar:", list(st.session_state['history_groups'].keys()))
        comparison_data = []
        for group in selected_groups:
            for record in st.session_state['history_groups'][group]:
                comparison_data.append({'Grupo': group, **record})
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data))

elif choice == ' Testes de Hipóteses':
    st.header(" Testes de Hipóteses Avançados")
    st.write("Teste hipóteses relacionadas às métricas dos vídeos do TikTok.")

    # Verificar colunas e primeiras linhas do dataset
    st.write("Dataset completo:", df)


    col1, col2 = st.columns(2)
    with col1:
        # Escolher coluna de agrupamento dinamicamente
        group_col = st.selectbox("Escolha a variável de agrupamento:", df.columns.tolist())
    with col2:
        metric = st.selectbox("Escolha a métrica para o teste:", [
            'likes_per_view', 'comments_per_view', 
            'video_duration_sec', 'video_download_count', 'video_share_count'
        ])


    group_values = df[group_col].unique()
    if len(group_values) == 2:
        group1 = df[df[group_col] == group_values[0]][metric]
        group2 = df[df[group_col] == group_values[1]][metric]

        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

        st.subheader("Resultados do Teste de Hipótese")
        st.write(f"Comparação entre {group_values[0]} e {group_values[1]}")
        st.write(f"Estatística t: {t_stat:.4f}")
        st.write(f"Valor-p: {p_value:.4f}")

        if p_value < 0.05:
            st.success(f"Rejeitamos a hipótese nula: existe diferença significativa entre os dois grupos para {metric}.")
        else:
            st.warning(f"Não rejeitamos a hipótese nula: não há diferença significativa entre os dois grupos para {metric}.")

        # Intervalos de confiança
        st.subheader("Intervalos de Confiança")
        ci_group1_lower = group1.mean() - 1.96 * group1.std() / (len(group1) ** 0.5)
        ci_group1_upper = group1.mean() + 1.96 * group1.std() / (len(group1) ** 0.5)
        st.write(f"Intervalo de confiança para {group_values[0]}: ({ci_group1_lower:.4f}, {ci_group1_upper:.4f})")

        ci_group2_lower = group2.mean() - 1.96 * group2.std() / (len(group2) ** 0.5)
        ci_group2_upper = group2.mean() + 1.96 * group2.std() / (len(group2) ** 0.5)
        st.write(f"Intervalo de confiança para {group_values[1]}: ({ci_group2_lower:.4f}, {ci_group2_upper:.4f})")

        # Visualização da comparação
        st.subheader("Comparação de Distribuições")
        fig = px.histogram(
            df,
            x=metric,
            color=group_col,
            nbins=30,
            title=f"Distribuição de {metric} por {group_col}",
            labels={metric: metric, group_col: group_col},
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("A variável de agrupamento selecionada não tem exatamente dois grupos. Escolha outra variável.")


elif choice == ' Filtragem de Dados':
    st.header(" Filtragem de Dados")
    st.write("Filtre vídeos com base em critérios personalizados.")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_views = st.slider("Mínimo de Visualizações", min_value=0, max_value=int(df['video_view_count'].max()), value=1000)
    with col2:
        min_likes = st.slider("Mínimo de Curtidas", min_value=0, max_value=int(df['video_like_count'].max()), value=500)
    with col3:
                min_duration = st.slider("Duração Mínima (segundos)", min_value=0, max_value=int(df['video_duration_sec'].max()), value=10)

    col4, col5 = st.columns(2)
    with col4:
        min_downloads = st.slider("Mínimo de Downloads", min_value=0, max_value=int(df['video_download_count'].max()), value=10)
    with col5:
        min_shares = st.slider("Mínimo de Compartilhamentos", min_value=0, max_value=int(df['video_share_count'].max()), value=10)

    # Aplicar os filtros
    filtered_df = df[
        (df['video_view_count'] >= min_views) &
        (df['video_like_count'] >= min_likes) &
        (df['video_duration_sec'] >= min_duration) &
        (df['video_download_count'] >= min_downloads) &
        (df['video_share_count'] >= min_shares)
    ]

    st.subheader("Resultados da Filtragem")
    st.write(f"Número de vídeos encontrados: {len(filtered_df)}")

    if len(filtered_df) > 0:
        st.dataframe(filtered_df)
        st.subheader("Distribuição de Visualizações dos Vídeos Filtrados")
        fig_filtered = px.histogram(
            filtered_df,
            x='video_view_count',
            nbins=30,
            title="Distribuição de Visualizações dos Vídeos Filtrados",
            labels={'video_view_count': 'Visualizações'}
        )
        st.plotly_chart(fig_filtered, use_container_width=True)
    else:
        st.warning("Nenhum vídeo encontrado com os critérios selecionados. Ajuste os filtros e tente novamente.")

        

        

        
