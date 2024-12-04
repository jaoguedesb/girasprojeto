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
def perform_regression(df, x_col, y_col):
    """Realiza uma regressão linear simples usando statsmodels."""
    X = df[x_col]
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
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 30px;
    }
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
    # Calcular a matriz de correlação com valores absolutos
    corr = df[['video_view_count', 'video_like_count', 'video_comment_count', 
               'video_duration_sec', 'video_download_count', 'video_share_count']].corr().abs()
    # Plotar o mapa de correlação
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Mapa de Correlação Entre Métricas (0 a 1)",
        labels=dict(color="Correlação (|r|)")
    )
    st.plotly_chart(fig_corr, use_container_width=True)

elif choice == ' Melhores Vídeos':
    st.header(" Melhores Vídeos")
    st.write("Identifique os vídeos mais populares com base no número de visualizações.")

    # Selecionar os 10 vídeos mais populares
    top_videos = df.sort_values(by='video_view_count', ascending=False).head(10)

    # Gráfico de dispersão (bubble chart)
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
            'video_share_count': 'Número de Compartilhamentos'
        },
        hover_data=['video_id']
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
    top_videos_display = top_videos[['video_id', 'video_view_count', 'video_like_count', 'video_comment_count']]
    top_videos_display.columns = ['ID do Vídeo', 'Visualizações', 'Curtidas', 'Comentários']
    st.dataframe(top_videos_display)

elif choice == ' Modelos de Regressão':
    st.header(" Modelos de Regressão Linear")
    st.write("Analise como diferentes métricas influenciam as visualizações dos vídeos.")

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Selecione a variável independente (X):", [
            'video_like_count', 'video_comment_count', 
            'video_duration_sec', 'video_download_count', 'video_share_count'
        ])
    with col2:
        y_col = 'video_view_count'
        st.write(f"Variável dependente (Y): {y_col}")

    # Treinar o modelo de regressão
    model = perform_regression(df, x_col, y_col)
    st.subheader(f"Resumo do Modelo de Regressão (Y: {y_col}, X: {x_col})")
    st.text(model.summary())

    st.subheader("Visualização do Modelo")
    df['prediction'] = model.predict(sm.add_constant(df[x_col]))
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title="Regressão Linear: Valores Observados vs. Preditos",
        labels={x_col: x_col, y_col: y_col},
        trendline="ols"
    )
    fig.add_scatter(x=df[x_col], y=df['prediction'], mode='lines', name='Linha de Regressão')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Previsão com Base no Modelo")
    input_value = st.number_input(f"Insira um valor para {x_col}:", min_value=0.0, step=1.0)
    if st.button("Prever"):
        prediction = model.predict([1, input_value])[0]
        st.success(f"Previsão de visualizações: {prediction:.2f}")

elif choice == ' Testes de Hipóteses':
    st.header(" Testes de Hipóteses")
    st.write("Teste hipóteses relacionadas às métricas dos vídeos do TikTok.")

    st.subheader("Escolha os Parâmetros para o Teste de Hipótese")
    metric = st.selectbox("Escolha a métrica para o teste:", [
        'likes_per_view', 'comments_per_view', 
        'video_duration_sec', 'video_download_count', 'video_share_count'
    ])
    pop_mean = st.number_input(f"Média populacional esperada para {metric}:", min_value=0.0, step=0.01, value=0.1)

    if st.button("Executar Teste de Hipótese"):
        sample_data = df[metric]
        t_stat, p_value = ttest_1samp(sample_data, pop_mean)

        st.subheader("Resultados do Teste de Hipótese")
        st.write(f"**Métrica:** {metric}")
        st.write(f"**Média da Amostra:** {sample_data.mean():.4f}")
        st.write(f"**Média Populacional Esperada:** {pop_mean}")
        st.write(f"**Estatística t:** {t_stat:.4f}")
        st.write(f"**Valor-p:** {p_value:.4f}")

        if p_value < 0.05:
            st.success(f"Rejeitamos a hipótese nula: a média de {metric} é significativamente diferente de {pop_mean}.")
        else:
            st.warning(f"Não rejeitamos a hipótese nula: a média de {metric} não é significativamente diferente de {pop_mean}.")

    st.subheader("Distribuição da Métrica")
    fig = px.histogram(df, x=metric, nbins=30, title=f"Distribuição de {metric}")
    st.plotly_chart(fig, use_container_width=True)

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
