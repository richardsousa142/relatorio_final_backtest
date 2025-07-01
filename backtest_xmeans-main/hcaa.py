'''
HCAA SIMPLIFICADO, SEM LIMIT ON GROWTH
'''
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hr
from scipy.cluster.hierarchy import dendrogram, fcluster
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform

class Tree:
  '''
  Classe Tree serve para criar a árvore que será usada no estágio adicional e nos ajudará a
  atribuir peso para os clusters e ativos contidos nesses clusters.

  Parameters
  ---------------
  value: int
          É o valor que será guardado por nó, no nosso caso esse valor guardará o indice do cluster criado
          ou o indice dos ativos, que são as folhas.
  weight: float
          É o peso que será atribuido a cada cluster ou ativo
  '''
  def __init__(self, value, weight = None) -> None:
      self.left = None
      self.right = None
      self.data = value
      self.weight = weight

def get_stocks(asset, s_date, e_date):
  '''
    Essa função como o próprio nome já diz tem como objetivo pegar os dados dos ativos que iremos usar
    para performar todo o processo de criação do portfolio.

    Para obter esses dados usamos a API de finanças do Yahoo, yfinance

    Parameters
    ------------
    asset: list
           Lista de strings onde cada string é o ticker do ativo que desejamos obter os dados
    s_date: str
            É a string que usamos para informar a data de inicio para obter os dados, ou seja, queremos
            que os dados obtidos seja a partir dessa data.
    e_date: str
            Semelhante ao s_date, e_date é a data final que usamos para informar até o momento que
            queremos os dados.

    return
    ------------
    data : dataframe pandas
            Dataframe contendo todos os dados dos ativos que passamos como parametro para a API
  '''
  data = yf.download(asset, start = s_date, end = e_date)['Adj Close']
  return data

def get_correlation(data):
  '''
  Obtem a correlação da matriz data, para obter a correlação usamos o metodo  Pearson

  Parameters
  ------------
  data: dataframe pandas
          É a matriz obtida através da API do Yahoo e contem todos os dados dos ativos passados

  Return
  ------------
  data.corr : dataframe pandas
          Matriz contendo a correlação entre os ativos
  '''
  return data.corr(method='pearson')

def calc_distance(correlation):
  ''''
  O processo de hierarchical clustering precisa de uma medida de distance, então para isso
  usaremos a medida de Mantegna 1999.

  Di,j = √0.5 * (1 - Pi,j) *P = representação de rho

  Parameters
  ------------
  correlation: dataframe pandas
      Dataframe contendo a correlação entre os ativos

  Return
  ------------
  distance: ndarray
      Matriz contendo a distancia calculada com base na correlação fornecida
  '''
  distance = np.sqrt(0.5 * (1 - correlation))
  return distance

def euclidean_distance_improve(len_stocks, distance_df):
  eucli_dist = pd.DataFrame()
  for n in range(len_stocks):
    for k in range(( distance_df.shape[0] * distance_df.shape[1] )):
      row = int(np.floor(k / distance_df.shape[1]))
      col = k % distance_df.shape[1]
      eucli_dist.loc[row, col] = np.sqrt((distance_df.iloc[n, row] - distance_df.iloc[n, col]) ** 2)
  eucli_dist.index = distance_df.index
  eucli_dist.columns = distance_df.columns
  return eucli_dist

def hierarchical_clustering(euclidean_distance, linkage):
  '''
  Performa o calculo da matriz de linkage usando a biblioteca scipy e o metodo linkage

  Parameters
  ------------
  euclidean_distance: ndarray
      Matriz contendo a distancia euclidiana
  linkage: str
      String para informar qual será o metodo de linkage utilizado

  Return
  ------------
  clustering_matrix: ndarray
      A matriz de hierarchical clustering codificada como uma matriz de link
  '''
  #condensed_distance = squareform(euclidean_distance)
  clustering_matrix = hr.linkage(euclidean_distance, method = linkage, optimal_ordering = True)
  return clustering_matrix

def get_idx_cluster_merged(clustering, len_stocks):
  '''
  Através da matriz de linkage podemos obter os indices dos clusters que foram combinados
  usando para isso a coluna 1 e 2 dessa matriz.

  Essa função retorna um dicionario onde a chave é um indice do cluster criado para comportar
  os dois outros clusters que foram combinados, já os valores são os clusters que foram
  combinados.

  Caso tenha um ponto de cutoff o indice que antes teria como valor dois clusters
  agora será formado portodas as folhas que pertenciam aos clusters anteriores ao indice em
  questão.

  Parameters
  -----------------
  clustering : ndarray
              É a matriz de linkage resultante do processo de hierarchical clustering
  len_stocks: int
              É o tamanho da lista contendo os ativos de interesse

  return
  -----------------
  cluster_merged: dict
              Dicionario contendo como chave os indices dos clusters que foram criados através
              da combinação de outros, e valores os indices dos clusters que foram envolvidos na
              criação do clusters
  '''
  cluster_merged = {
    len_stocks + i: [int(clustering[i, 0]), int(clustering[i, 1])]
    for i in range(len(clustering))
  }
  return cluster_merged

def create_tree_from_clusters(cluster, dicio, raiz, len_asset):
  '''
  Nessa funcao temos como objetivo pegar o dicionario e usar os dados contidos ali para criar a arvore.
  Para isso, recebemos os valores da ultima chave no parametro 'cluster', com isso verificamos o valor
  de cluster[0] >= 10, se for maior de 10 significa que foi um cluster criado no processo de linkage e
  portanto tem ativos ou clusters associados a ele. O mesmo vale para o cluster[1].

  Usamos cluster[0] para representar os ativos ou clusters que estão contidos na subarvore esquerda, e
  cluster[1] para representar os ativos ou clusters que estão contidos na subarvore direita.

  Tendo essas duas explicações prévias, começamos o codigo de fato.

  Se cluster[0] >= 10 criamos o nó esquerdo da arvore contendo esse valor, pois abaixo dele terá dois ativos
  ou outros clusters. O mesmo vale para o cluster[1].

  A função art_leaf_node sera chamada quando o valor contido em cluster[0] ou cluster[1] for uma lista

  A cada chamada recursiva da função passamos o novo valor de cluster, e também passamos a raiz.left ou
  raiz.right para que nas proximas execuções a arvore continue sendo criada

  Exemplo esquematico
  ------------------------------------------------------------------------------------------------
              [14, 17]
      [14]                  [17]
    [8]    [4]        [16]         [15]
                  [3]    [13]  [7]     [12]
                      [9]    [5]   [11]    [2]
                                [1]    [10]
                                    [6]    [0]

  Parameters
  ----------
  cluster : list
      Lista contendo dois elementos, esses elementos são os indices dos clusters que foram combinados na
      criação do ultimo cluster
  dicio : dict
      Dicionario onde a chave é o indice dos clusters que foram criados através da combinação de outros
      clusters, e os valores são os indices desses clusters
  raiz : Tree
      Arvore que iremos criar com base no dicionario e os clusters que sao seus valores

  Return
  ------
  cluster : list
      Lista contendo dois elementos ou apenas um, esses elementos são os indices dos clusters que
      foram combinados na criação do ultimo cluster
  '''
  for i, subcluster in enumerate(cluster):
    if subcluster < len_asset:
      if i == 0:
          raiz.left = Tree(subcluster)
      else:
          raiz.right = Tree(subcluster)
    else:
      next_cluster = dicio[subcluster]
      node = Tree(subcluster)
      if i == 0:
          raiz.left = node
      else:
          raiz.right = node
      create_tree_from_clusters(next_cluster, dicio, node, len_asset)

def weight_tree(arvore):
  '''
  Essa função é usada para dar peso para os ativos e para os cluster, como um dendrogram é uma arvore binaria
  criamos uma arvore e definimos o peso para os nós.

  Parameters
  ----------
  arvore : Tree
      Raiz da arvore

  Return
  --------------
  vet_weight : list
      vetor contendo o peso de cada folha da arvore
  '''
  if not arvore:
    return []

  stack = [(arvore, arvore.weight)]  # Pilha para percorrer a árvore com o peso atual
  vet_weight = []

  while stack:
    node, weight = stack.pop()

    # Define o peso do nó atual
    node.weight = weight

    # Se for folha, adiciona ao vetor de pesos
    if not node.left and not node.right:
      vet_weight.append(node.weight)
    else:
      # Adiciona os filhos à pilha com o peso atualizado
      if node.right:
          stack.append((node.right, weight / 2))
      if node.left:
          stack.append((node.left, weight / 2))

  return vet_weight

def main(data, asset):
  #asset = ['MSFT', 'PCAR', 'JPM', 'AAPL', 'GOOGL', 'AMZN', 'ITUB', 'VALE', 'SHEL', 'INTC']
  #start = '2016-01-01'; end = '2022-01-01'
  #data_stocks = get_stocks(asset, start,  end)

  # Stage 1: Hierarchical Clustering

  correlation = get_correlation(data)
  distance = calc_distance(correlation)
  distance_euclidean = (pdist(distance, metric='euclidean'))
  clustering = hierarchical_clustering(distance_euclidean, 'ward')
  
  # Etapa 2: Determinação dos clusters
  # Define o número de clusters desejado (exemplo: 2 clusters)
  num_clusters = 4
  clusters = fcluster(clustering, t=num_clusters, criterion='maxclust')
  #print(clusters)

  # Additional Stage to aux the weight stage

  #dicionario contendo os indices dos clusters gerados da combinação de outros 2
  cluster_merged = get_idx_cluster_merged(clustering, len(asset))
    
  #lista contendo as chaves do dicionario
  keys = list(cluster_merged.keys())
  
  #ultimos clusters merged
  cluster = cluster_merged[keys[-1]]
  
  #Cria arvore onde a raiz é os dois ultimos clusters combinados
  raiz = Tree(cluster, 100)
  
  #criando a arvore atravez dos dois ultimos clusters combinados
  #usando o dicionario para mapear os clusters que foram criados atraves da combinação
  #passando a arvore para prencher o campo data de cada no
  create_tree_from_clusters(cluster, cluster_merged, raiz, len(asset))
  
  #Stage 2: Assigning weights to clusters
  
  #vetor com o peso de cada ativo e cluster
  vet_weight = weight_tree(raiz)
  
  #dicionario mapeando o ativo e seu respectivo peso no portfolio
  dict_asset_weight = {key: f'{value}%' for key, value in zip(list(hr.leaves_list(clustering)), vet_weight)}
 
  #Print of dendrogram
  #print(vet_weight)
  #print(cluster_merged)
  
  #print(dict_asset_weight)
  #plt.figure(figsize=(6, 6))
  #print(cluster_merged)
  return np.array(vet_weight) / 100
