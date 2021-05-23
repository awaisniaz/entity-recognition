import pandas as pd
import numpy as np
from itertools import chain
class HandlingClass:
  def dataloader(self):
      data = pd.read_csv('ner_dataset.csv',encoding='unicode_escape')
      return data
  def get_dict_data(self,data, token_or_tag):
      tok2idx = {}
      idx2tok = {}
      if token_or_tag == 'token':
          vocab = list(set(data['Word'].to_list()))
      else:
          vocab = list(set(data['Tag'].to_list()))

      idx2tok = {idx:tok for idx,tok in enumerate(vocab)}
      tok2idx = {tok:idx for idx,tok in enumerate(vocab)}

      return  tok2idx,idx2tok





