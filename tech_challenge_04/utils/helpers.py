import pandas as pd
import numpy as np


def carregar_dados(caminho_arquivo):
  """Carrega um arquivo CSV e retorna um DataFrame do Pandas.

  Args:
    caminho_arquivo (str): Caminho completo do arquivo CSV.

  Returns:
    pandas.DataFrame: DataFrame contendo os dados do arquivo.
  """

  return pd.read_csv(caminho_arquivo)


def normalizar_dados(dados):
  """Normaliza os dados de um DataFrame.

  Args:
    dados (pandas.DataFrame): DataFrame a ser normalizado.

  Returns:
    pandas.DataFrame: DataFrame com os dados normalizados.
  """

  return (dados - dados.mean()) / dados.std()


def calcular_metrica(y_true, y_pred):
  """Calcula uma métrica de avaliação (por exemplo, precisão) entre os valores verdadeiros e preditos.

  Args:
    y_true (array-like): Valores verdadeiros.
    y_pred (array-like): Valores preditos.

  Returns:
    float: Valor da métrica calculada.
  """

  # Implementação da métrica desejada (por exemplo, precisão, recall, F1-score)
  from sklearn.metrics import accuracy_score
  return accuracy_score(y_true, y_pred)
