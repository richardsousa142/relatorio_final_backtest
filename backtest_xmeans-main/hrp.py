import yfinance as yf
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hr
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_stocks(asset, s_date, e_date):
  '''
    Essa função como o próprio nome já diz tem como objetivo pegar os dados dos ativos que iremos usar
    para performar todo o processo de criação do portfolio.

    Para obter esses dados usamos a API de finanças do Yahoo, yfinance

    Parameters
    ------------
    asset
         Lista de strings onde cada string é o ticker do ativo que desejamos obter os dados
    s_date
        É a string que usamos para informar a data de inicio para obter os dados, ou seja, queremos
        que os dados obtidos seja a partir dessa data.
    e_date
        Semelhante ao s_date, e_date é a data final que usamos para informar até o momento que
        queremos os dados.

    return
    ------------
    data
        Dataframe contendo todos os dados dos ativos que passamos como parametro para a API
    '''
  data = yf.download(asset, start = s_date, end = e_date)['Adj Close']
  return data

def get_correlation(data):
  '''
    Obtem a correlação da matriz data, para obter a correlação usamos o metodo  Pearson

    Parameters
    ------------
    data
        É a matriz obtida através da API do Yahoo e contem todos os dados dos ativos passados

    Return
    ------------
    data.corr
        Matriz contendo a correlação entre os ativos
    '''
  return data.corr(method='pearson')

def calc_distance(correlation):
  '''
  Função que tem como unico objetivo calcular e retornar a matriz de
  "correlation matrix distance". O retorno dessa função é usado para
  calcular a distancia euclidiana na função "euclidian_distance".


  Parameters
  ----------
  correlation
    é a matriz de correlação "correlation matrix distance"

  Return
  ------
  distance
    é a matrix de correlação
  '''
  distance = np.sqrt(0.5 * (1 - correlation))
  return distance

def euclidian_distance_improve(len_stocks, distance):
  '''
  Função que tem como unico objetivo calcular e retornar a distancia
  euclidiana que sera usada para fazer a etapa de "Quasi-Diagonalisatio"
  do portfolio.

  Esse modo de calcular a distancia euclidiana foi inspirado no artigo
  https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1609991&dswid=-2657
  No artigo os autores recomendam calcular a distancia euclidiana entre as
  colunas

  Parameters
  ----------
  distance
    é a matriz de correlação distancia
  len_stocks
    tamanho do vetor de tickers ("ações") que tiveram dados baixados

  Return
  -------
  eucli_dist
    é a matriz com todas as distancias euclidianas calculadas
  '''
  eucli_dist = pd.DataFrame(index=distance.index, columns=distance.columns)
  for n in range(len_stocks): #algoritmo quadratico
    for k in range(( distance.shape[0] * distance.shape[1] )):
      row = int(np.floor(k / distance.shape[1]))
      col = k % distance.shape[1]
      eucli_dist.iloc[row][col] = np.sqrt((distance.iloc[n][row] - distance.iloc[n][col])**2)
  return eucli_dist

def hierarchical_clustering(euclidean_distance, linkage):
  '''
  Função para criar os clusters com base na matriz fornecida e no metodo de link

  Parameters
  -----------
  euclidean_distance
    é a matriz com as distancias calculadas entre os ativos
  linkage
    é o metodo que será utilizadao para realizar o link dos clusters

  Return
  ------
  clustering_matrix
  é o cluster criado com base na matriz e no metodo fornecidos
  '''
  #linkage matrix
  clustering_matrix = hr.linkage(euclidean_distance, method = linkage, optimal_ordering = True)
  return clustering_matrix

def quasi_diagonalization(linkage_matrix, original_order):
  '''
  Organizar a linkage_matriz para colocar os clusters na diagonal principal

  Parameters
  -----------
  linkage_matrix
    é a matriz resultante do processo de clusterização
  original_order
    é a ordem original dos ativos no vetor asset da main

  Return
  ------
  quasi_diag_link_matrix
    é a matriz com a diagonal principal contendo os clusters
  mapping
    é a ordem das folhas do cluster
  '''
  order = hr.leaves_list(linkage_matrix)
  quasi_diag_link_matrix = linkage_matrix[order-1] #NAO SEI SE PODE SUBTRAIR 1 AQUI, PESQUISAR DPS

  mapping = {original_order[i]: order[i] for i in range(len(order))}
  return quasi_diag_link_matrix, mapping

'''
Funcao retirada diretamente do site do Lopéz de Prado onde tem o codigo do HRP
disponivel
'''
def recursive_bisection(covariance_matrix, sortIx):
  #create a column vector w with indices equal to sortIx and all values equal to 1
  w = pd.Series(1, index=sortIx)
  #create a list cItems containing one element, which corresponds to the list sortIx
  cItems = [sortIx]
  while len(cItems) > 0:
    #divide o vetor cItems em 2 tendo como ponto de  divisao o meio do vetor
    cItems=[ i[j: k] for i in cItems for j, k in ( (0, len(i) // 2), (len(i) // 2, len(i)) ) if len(i) > 1 ]
    for i in range(0, len(cItems), 2):
      cItems0 = cItems[i]   # cluster 1
      cItems1 = cItems[i+1] # cluster 2
      #calcula a variança e variança inversa do cluster fonecido por parametro
      cVar0 = getClusterVar(covariance_matrix, cItems0)
      cVar1 = getClusterVar(covariance_matrix, cItems1)
      #calcula o alfa para ser calculado o peso
      alpha = 1 - cVar0 / (cVar0+cVar1)
      #atualiza o peso
      w[cItems0] *= float(alpha) # weight 1
      w[cItems1] *= float(1-alpha) # weight 2
  #retorna o vetor de peso para o portfolio
  return w

'''
Funcao retirada diretamente do site do Lopéz de Prado onde tem o codigo do HRP
disponivel
'''
def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_= cov.loc[cItems, cItems] # matrix slice
    w_= getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar

'''
Funcao retirada diretamente do site do Lopéz de Prado onde tem o codigo do HRP
disponivel
'''
def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

'''
Funcao retirada diretamente do site do Lopéz de Prado onde tem o codigo do HRP
disponivel
'''
def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0],link[-1, 1]])
    numItems=link[-1, 3] # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2) # make space
        df0 = sortIx[sortIx >= numItems] # find clusters
        i = df0.index; j = df0.values-numItems
        sortIx[i] = link[j, 0] # item 1
        df0 = pd.Series(link[j, 1],index=i+1)
        sortIx = pd.Series(np.append(sortIx.values, df0))
        sortIx = sortIx.sort_index() # re-sort
        sortIx.index = range(sortIx.shape[0]) # re-index
    return sortIx.tolist()

'''
A forma de calcular o peso dos ativos usando recursive bisection tem um problema
que é, a ordem dos ativos importam, portanto, caso ocorra uma alteração na ordem
os pesos serão diferentes e isso ocorre pq esse metodo não utiliza a linkage ma
trix e sim somente a matriz de covariança, para resolver esse problema foram
criados algoritmos que usam a linkage matrix para que o peso dos ativos sejam
preservados mesmo que ocorra alteração na ordem dos mesmo, são eles 'bottom-up e
'top-down'.
'''

def main(data):
  #Para ações brasileiras, usar: TICKER.SA
  #asset = ['MSFT', 'PCAR', 'JPM', 'AAPL', 'GOOGL', 'AMZN', 'ITUB', 'VALE', 'SHEL', 'INTC']
  #start = '2016-01-01'; end = '2022-01-01'
  #data_stocks = get_stocks(asset, start,  end)
  # Stage 1: Tree clustering
  correlation = get_correlation(data)
  distance = calc_distance(correlation)
  distance_euclidian_square = ((pdist(distance, metric='euclidean')))
  clustering = hierarchical_clustering(distance_euclidian_square, 'ward')
  # Stage 2: Quasi-Diagonalisation
  sortIx = getQuasiDiag(clustering)
  sorted_assets = [data.columns[i] for i in sortIx]
  #sortIx = hr.leaves_list(clustering).tolist()
  #sortIx = correlation.index[sortIx].tolist() # recover labels

  # Stage 3: Recursive Bisection
  rec_bisection = recursive_bisection(data.cov(), sorted_assets)
  
  #plot_dendrogram_figure([10, 5], clustering, data_stocks)
  #plot_network(data_stocks)
  return rec_bisection.values