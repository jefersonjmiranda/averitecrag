import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from itertools import combinations

def printText(t):
  st.markdown(
    f"""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
      <p style='color: #666666;'>{t}</p>
    </div>
    """,
    unsafe_allow_html=True
  )
  st.write('\n')
  
def show():

  docs = {}
  with open('docs.pkl', 'rb') as f:
    docs = pickle.load(f)

  averitec = pd.read_json('data_train.json')
  df = averitec
  
  tab1, tab2 = st.tabs(["Avaliação", "Visualização"])

  with tab1:
    st.header('Dataset AVeriTeC')
    st.write('Disponível em: https://fever.ai/dataset/averitec.html')
    st.write('Artigo: https://proceedings.neurips.cc/paper_files/paper/2023/file/cd86a30526cd1aff61d6f89f107634e4-Paper-Datasets_and_Benchmarks.pdf')
    
    st.header('Descrição')
    printText("""Este dataset é para uso em automatização de checagem de fatos.
                Está em inglês e contém alegações de cunho político com uma alegação (claim) e sua classificação (label),
                que pode ser de um dos quatro tipos: Supported, Refuted, Conflicting Evidence/Cherrypicking ou Not Enough Evidence.
                Cada classificação é acrescida de uma justificativa (justification).
                Adicionalmente, cada alegação pode conter N perguntas (questions).
                Cada pergunta pode conter mais de um responta. Esta resposta pode ser classificada por um ou mais desses tipos:
                Event/Property Claim, Numerical Claim, Causal Claim, Quote Verification e/ou Position Statement.""")
    
    st.header('Abstract') 
    printText(
      """
        Existing datasets for automated fact-checking have substantial limitations, such as relying on artificial claims, lacking annotations for evidence and intermediate reasoning, or including evidence published after the claim. In this paper we introduce AVeriTeC, a new dataset of 4,568 real-world claims covering fact-checks by 50 different organizations. Each claim is annotated with question-answer pairs supported by evidence available online, as well as textual justifications explaining how the evidence combines to produce a verdict. Through a multi-round annotation process, we avoid common pitfalls including context dependence, evidence insufficiency, and temporal leakage, and reach a substantial inter-annotator agreement of κ=0.619 on verdicts. We develop a baseline as well as an evaluation scheme for verifying claims through several question-answering steps against the open web.
      """)
    
    st.header('Formato dos Dados')
    printText('Descrição de cada coluna, texto em inglês extraído diretamente da fonte:')
    st.html("""
      <b>claim:</b> The claim text itself.<br>
      
      <b>required_reannotation:</b> True or False. Denotes that the claim received a second round of QG-QA and quality control annotation.<br>
      
      <b>label:</b> The annotated verdict for the claim.<br>
      
      <b>justification:</b> A textual justification explaining how the verdict was reached from the question-answer pairs.<br>
      
      <b>claim_date:</b> Our best estimate for the date the claim first appeared.<br>
      
      <b>speaker:</b> The person or organization that made the claim, e.g. Barrack Obama, The Onion.<br>
      
      <b>original_claim_url:</b> If the claim first appeared on the internet, a url to the original location.<br>
      
      <b>cached_original_claim_url:</b> Where possible, an archive.org link to the original claim url.<br>
      
      <b>fact_checking_article:</b> The fact-checking article we extracted the claim from.<br>
      
      <b>reporting_source:</b> The website or organization that first published the claim, e.g. Facebook, CNN.<br>
      
      <b>location_ISO_code:</b> The location most relevant for the claim. Highly useful for search.<br>
      
      <b>claim_types:</b> The types of the claim.<br>
      
      <b>fact_checking_strategies:</b> The strategies employed in the fact-checking article.<br>
      
      <b>questions:</b><br>
      
      &emsp;<b>question:</b> A fact-checking question for the claim.<br>
      
      &emsp;<b>answers:</b><br>
      
      &emsp;&emsp;<b>answer:</b> The answer to the question.<br>
      
      &emsp;&emsp;<b>answer_type:</b> Whether the answer was abstractive, extractive, boolean, or unanswerable.<br>
      
      &emsp;&emsp;<b>source_url:</b> The answer to the question.<br>
      
      &emsp;&emsp;<b>cached_source_url:</b> An archive.org link for the source url.<br>
      
      &emsp;&emsp;<b>source_medium:</b> The medium the answer appeared in, e.g. web text, a pdf, or an image.<br>
      """)
    
    st.html('<hr style="border: 1px solid #2F4F4F; border-radius: 5px;">')
    
    st.header("Valores Nulos")
    printText("Aqui vemos que algumas colunas que deveriam ser importantes para a checagem de fatos, como original_claim_url ou reporting_source, contém valores nulos.")
    null_counts = averitec.isnull().sum()
    st.write(null_counts)
    
    st.html('<hr style="border: 1px solid #2F4F4F; border-radius: 5px;">')
    
    st.header("Classes (labels)")
    printText("As classes indicam como uma alegação pode ser classificada, com base no texto de referência.")
    exploded_df = averitec.explode('label')
    unique_claim_types = exploded_df['label'].unique()
    st.write(unique_claim_types)
    label_counts = exploded_df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    fig = px.bar(label_counts, x='label', y='count', title='Histograma das Classes')
    st.plotly_chart(fig)

    st.html('<hr style="border: 1px solid #2F4F4F; border-radius: 5px;">')
    
    st.header("Divergência entre Anotadores (required_reannotation)")
    printText("""
              Do artigo:
              \"We passed the remainder through a five-phase pipeline – see Figure 1.
              First, an annotator extracts claims and relevant metadata from each article, providing context independence.
              Second, an annotator generates questions and answers them using the web.
              These annotators also choose a temporary verdict.
              Third, a different annotator provides a justification and a verdict based solely
              on the annotated question-answer pairs;
              this serves as an evidence sufficiency check.
              Any claim for which the two verdicts do not match is passed through the last two phases again.
              If the verdicts still disagree, the claim is discarded.
              Different annotators were used for each claim in each phase – i.e., no annotator saw the same claim twice.\"
              """)
    st.write(f"Proporção de alegações não reanotadas: {2206/(2206+862)}")
    exploded_df = averitec.explode('required_reannotation')
    label_counts = exploded_df['required_reannotation'].value_counts().reset_index()
    label_counts.columns = ['required_reannotation', 'count']
    st.write(label_counts)
    fig = px.bar(label_counts, x='required_reannotation', y='count', title='Histograma da Reanotação')
    st.plotly_chart(fig)

    st.html('<hr style="border: 1px solid #2F4F4F; border-radius: 5px;">')

    st.header("Tipos de Alegação")
    printText("""
              Do artigo:
              \"We also annotate the claim type and fact-checking strategy of each claim.
              Type represents common aspects, e.g., whether claims are about numerical facts;
              strategy represents the approach of the fact-checkers, e.g.,
              whether they relied on expert testimony.
              Types andstrategies should not be used as input to models (at inference time),
              but can provide useful data for analysis.\"
              """)
    df_exploded = averitec.explode('claim_types')
    claim_counts = df_exploded['claim_types'].value_counts().reset_index()
    claim_counts.columns = ['claim_type', 'count']
    st.write("Contagem de tipos de alegação (individuais):")
    st.write(claim_counts)
    fig = px.bar(claim_counts, x='claim_type', y='count', title='Histograma dos Tipos de Alegação')
    st.plotly_chart(fig)
    
    tuple_counts = df['claim_types'].apply(tuple).value_counts().reset_index()
    tuple_counts.columns = ['claim_tuple', 'count']
    st.write("Contagem de tuplas de alegações:")
    st.write(tuple_counts)
    fig_tuples = px.bar(tuple_counts, x=tuple_counts.index, y='count', title='Histograma das Tuplas de Alegações')
    fig_tuples.update_layout(
      xaxis_title='Índice da Tupla',
      yaxis_title='Contagem',
      xaxis=dict(
        tickmode='array',
        tickvals=list(range(len(tuple_counts))),
        ticktext=[str(t) for t in tuple_counts['claim_tuple']]
      )
    )
    st.plotly_chart(fig_tuples)
    
    selected_tuple = st.selectbox('Selecione uma tupla de claim_types', options=tuple_counts['claim_tuple'])
    filtered_df = averitec[averitec['claim_types'].apply(tuple) == selected_tuple]
    st.write(f"Linhas do DataFrame com claim_types igual a {selected_tuple}:")
    st.write(filtered_df)
    
    st.html('<hr style="border: 1px solid #2F4F4F; border-radius: 5px;">')
    
    st.header("Tipos de Respostas")
    printText("""
              Para cada pergunta, pode haver tipos distintos de answer_types. Do artigo:
              \"Questions may have multiple answers, a natural way to show potential disagreements in the evidence.
                Questions can refer to previous questions, allowing for multi-hop reasoning.
                Answers (other than \"No answer could befound.\") must be supported by a source url linking to a web document.
              """)
    
    def extract_answer_types(questions):
      answer_types = []
      for question in questions:
          for answer in question['answers']:
              answer_types.append(answer['answer_type'])
      return answer_types

    df['all_answer_types'] = df['questions'].apply(extract_answer_types)
    exploded_answer_types = df.explode('all_answer_types')
    unique_answer_types = exploded_answer_types['all_answer_types'].unique()
    st.write(unique_answer_types)
    
    printText("Pode haver mais de um tipo de reposta para cada pergunta.")
    
    def get_answer_type_combinations(json_data):
        questions = json_data
        all_combinations = set()
        for question in questions:
            answer_types = list(set(answer['answer_type'] for answer in question['answers']))
            for i in range(1, len(answer_types) + 1):
                all_combinations.update(combinations(answer_types, i))
        return all_combinations
    
    all_combinations = set()
    for json_data in df['questions']:
        all_combinations.update(get_answer_type_combinations(json_data))

    all_combinations = sorted(all_combinations)

    selected_combination = st.selectbox(
        'Selecione uma combinação de answer_type',
        all_combinations
    )

    def contains_selected_combination(json_data, selected_combination):
        questions = json_data
        for question in questions:
            answer_types = {answer['answer_type'] for answer in question['answers']}
            if set(selected_combination).issubset(answer_types):
                return True
        return False

    df_filtered = df[df['questions'].apply(contains_selected_combination, selected_combination=selected_combination)]

    st.write(df_filtered)
    
    st.html('<hr style="border: 1px solid #2F4F4F; border-radius: 5px;">')
    
    st.header("Datas de Registro das Alegações")
    printText("Além de existirem valores nulos, muitas datas não fazem sentido.")
    
    def plot_dates(df):
      min_date = min(df['date'])
      max_date = max(df['date'])
      st.subheader('Filtrar por Data')
      start_date = st.date_input('Data de Início', min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
      end_date = st.date_input('Data de Fim', min_value=min_date.date(), max_value=max_date.date(), value=pd.to_datetime("2000-01-01"))

      start_date = pd.to_datetime(start_date)
      end_date = pd.to_datetime(end_date)

      filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

      counts = filtered_df['year_month'].value_counts().sort_index()

      fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        labels={'x': 'Data (Ano-Mês)', 'y': 'Frequência'},
        title='Histograma de Frequência por Mês'
      )

      fig.update_layout(xaxis_title='Data (Ano-Mês)', yaxis_title='Frequência')
      fig.update_xaxes(tickangle=90)

      st.plotly_chart(fig)
      
      return filtered_df

    df = averitec
    df['date'] = pd.to_datetime(df['claim_date'], format='%d-%m-%Y')
    df['year_month'] = df['date'].dt.to_period('M')
    date_filtered_df = plot_dates(df)
    date_filtered_df = date_filtered_df[['claim', 'claim_date']]
    st.table(date_filtered_df)
    
  with tab2:
    st.header("Exemplo")
    index = st.number_input(f'Linha (0..{len(averitec)}):', min_value=0, max_value=len(averitec), value=0)
    st.write(f"Linha selecionada: {index}")
    line = averitec.iloc[index]
    lidx = 0
    limit = len(line.index)-1
    for coluna in line.index:
      st.markdown(f"**{coluna}**:")
      if coluna=='questions':
        st.json(f"{line[coluna]}")
      else:
        st.write(f"{line[coluna]}")
      if lidx < limit:
        st.markdown("---")
      lidx+=1

    st.title('Text')
    st.write(docs[line['fact_checking_article']][0].page_content)
