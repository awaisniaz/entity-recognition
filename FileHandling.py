import pandas as pd
import numpy as np
from itertools import chain
class HandlingClass:
  def dataloader(self):
      data = pd.read_csv('ner_dataset.csv',encoding='unicode_escape')
      print(data.head(3))

