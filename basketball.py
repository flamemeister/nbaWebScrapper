import streamlit as st
import pandas as pd
import base64
import altair as alt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import requests

st.title('Статистика игроков НБА')

st.markdown("""
Простой тул который реализирует вебскрапинг информации о статистики игроков НБА
* **Использованные:** base64, pandas, streamlit, requests
* **Источник данных:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('Функции ввода')
selected_year = st.sidebar.selectbox('Год', list(reversed(range(1950, 2024))))

@st.cache_data
def load_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    response = requests.get(url, verify=False)
    html = pd.read_html(response.text, header=0)
    df = html[0]
    raw = df[df['Age'] != 'Age'].fillna(0)
    raw = raw.fillna(0)
    raw['Age'] = pd.to_numeric(raw['Age'], errors='coerce')
    raw = raw.dropna(subset=['Age'])
    numeric_columns = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA',
                       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    raw[numeric_columns] = raw[numeric_columns].apply(pd.to_numeric, errors='coerce')
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats

def display_player_stats(df):
    st.header('Показать статистику игрока выбранной команды')
    st.write('Измерение данных: ' + str(df.shape[0]) + ' ряд и ' + str(df.shape[1]) + ' колнны.')
    st.dataframe(df)

    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Скачатьь CSV файл</a>'
        return href

    st.markdown(filedownload(df), unsafe_allow_html=True)

def correlation_heatmap(dataframe):
    st.header('Тепловая карта матрицы интеркорелляции')

    df = dataframe.select_dtypes(include=[np.number])

    if not df.empty:
        corr = df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot()
    else:
        st.warning("No numeric data available for correlation heatmap.")


playerstats = load_data(selected_year)

st.sidebar.header('Функции ввода')

sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Комнда', sorted_unique_team, sorted_unique_team)

unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Позиция', unique_pos, unique_pos)

df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

display_player_stats(df_selected_team)

if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()
