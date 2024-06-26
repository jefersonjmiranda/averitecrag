import streamlit as st
from pages import analysis, answers

st.title('Dataset AVeriTeC - RAG')

page = st.sidebar.selectbox('Escolha uma página:', ['Análise Exploratória', 'Perguntas e Respostas'])

if page == 'Análise Exploratória':
  analysis.show()
elif page == 'Perguntas e Respostas':
  answers.show()
