import streamlit as st
import pandas as pd
import pickle
import json
import pandas as pd
import re
from typing import List
from nltk import ngrams

def highlightText(text, subtext):
  return text.replace(subtext, f"<span style='background-color: #FFFF00'>{subtext}</span>")

def applyHighlight(text, ngrams_data):
  final_text = text
  for n in ngrams_data.keys():
    ng = ngrams_data[n]
    for tuple in ng:
      tuple_text = ' '.join(tuple)
      final_text = highlightText(final_text, tuple_text) 
  return final_text

def createNGrams(text, subtext):
  ngrams_data = {}
  for n in range(10, 2, -1):
    ngram_result = list(ngrams(text.split(), n))
    ngram_sub_result = list(ngrams(subtext.split(), n))
    unique_ngrams = []
    for ngram in ngram_result:
      for sub_ngram in ngram_sub_result:
        if ngram==sub_ngram:
          if not ngram in unique_ngrams:
            unique_ngrams.append(ngram)
    ngrams_data[str(n)] = unique_ngrams
  return ngrams_data

def extract_json_content(text):
  result = []
  pattern = r'["\']label["\']:\s*["\']([^"\']+)["\']'
  matches = re.findall(pattern, text)
  for match in matches:
      result.append(match)
  return result

def show():

  docs = {}
  with open('docs.pkl', 'rb') as f:
    docs = pickle.load(f)

  averitec = pd.read_json('data_train.json')
  answers = pd.read_csv('answers.csv')
  classification = pd.read_json('classification.json')

  #-----------------------------------------------------------------------------
  # sidebar
  #-----------------------------------------------------------------------------

  size = len(answers)
  claim_options = list(averitec['claim'].head(size))

  claim_dict = {}
  for i, c in enumerate(claim_options):
    claim_dict[c] = i
    
  claim = st.selectbox(
    'Selecione uma afirmativa:',
    claim_options
  )

  index = claim_options.index(claim) | 0


  #-----------------------------------------------------------------------------
  # tabs
  #-----------------------------------------------------------------------------

  tab1, tab2, tab3 = st.tabs(["Claim", "Answers AVeriTeC", "Answers RAG"])

  with tab1:
    line = averitec.iloc[index]
    line_rag = answers.iloc[index]

    st.header('Clain Number '+str(index))
    st.write(claim)

    claim_types = line['claim_types']
    st.header('Claim Types')
    st.write(claim_types)
    
    fact_checking_strategies = line['fact_checking_strategies']
    st.header('Fact Checking Strategies')
    st.write(fact_checking_strategies)
    
    label = line['label']
    st.header('Label')
    st.write(label)

    required_reannotation = line['required_reannotation']
    st.header('Required Reannotation')
    st.write(required_reannotation)

    justification = line['justification']
    st.header('Justification')
    st.write(justification)
    
    st.title('Text')
    st.write(docs[line['fact_checking_article']][0].page_content)

  with tab2:
    st.header('Question / Answer from AVeriTeC dataset')

    for q_index, q in enumerate(line['questions']):
      q_text = 'QA ' + str(q_index+1)
      st.subheader(q_text)
      st.write(q)
      
    valid_json = line_rag['answers'].replace('\'{', '{').replace('}\'', '}').replace('\\"', '').replace('\\', '')
      
    answers = json.loads(valid_json)
    st.write(len(answers))
      
  with tab3:
    
    line_averitec = averitec.iloc[index]
    line = classification.iloc[index]

    text = docs[line_averitec['fact_checking_article']][0].page_content
    ngrams_data = createNGrams(text, line['output'])

    final_answer = applyHighlight(line['output'], ngrams_data)
    final_text = applyHighlight(text, ngrams_data)

    st.header('From Dataset')
    st.markdown(f"**claim**: {line_averitec['claim']}")
    st.markdown(f"**label**: {line_averitec['label']}")
    st.markdown(f"**justification**: {line_averitec['justification']}")

    st.header('From RAG')
    same_label = extract_json_content(line['output'])[0]==line_averitec['label'] 
    if same_label:
      st.html(f"<b>label</b>: <span style='background-color: #98FB98'>{extract_json_content(line['output'])[0]}</span>")
    else:
      st.html(f"<b>label</b>: <span style='background-color: #F08080'>{extract_json_content(line['output'])[0]}</span>")
    st.markdown("**output**:")
    st.markdown(final_answer, unsafe_allow_html=True)
    
    st.header('Reference (fact_checking_article)')
    st.markdown(final_text, unsafe_allow_html=True)
