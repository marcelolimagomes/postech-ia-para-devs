import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np


def gerar_resumo(texto):
  """Gera um resumo do texto utilizando a técnica de sumarização extractiva.

  Args:
      texto (str): Texto a ser resumido.

  Returns:
      str: Resumo do texto.
  """

  # Pré-processamento
  stop_words = set(stopwords.words('portuguese'))
  sentences = sent_tokenize(texto)
  word_embeddings = {}

  # Criar uma matriz de similaridade entre as frases
  for i, sent in enumerate(sentences):
    words = nltk.word_tokenize(sent)
    words = [w for w in words if not w in stop_words]
    word_embeddings[i] = words

  similarity_matrix = np.zeros((len(sentences), len(sentences)))
  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i != j:
        similarity_matrix[i][j] = cosine_distance(word_embeddings[i], word_embeddings[j])

  # Selecionar as frases mais importantes
  sentence_similarity_graph = np.array(similarity_matrix)
  sentence_similarity_graph[sentence_similarity_graph < (1 - (1 / len(sentence_similarity_graph)))] = 0
  sentence_indices = np.argsort(-np.sum(sentence_similarity_graph, axis=0))[:len(sentences) // 2]
  summary = ' '.join([sentences[i] for i in sentence_indices])
  return summary


# Exemplo de uso:
if __name__ == "__main__":
  texto = "Este é um exemplo de texto. Ele será utilizado para gerar um resumo. O resumo deve ser conciso e informativo."
  resumo = gerar_resumo(texto)
  print(resumo)
