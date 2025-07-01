import yfinance as yf
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hr
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import minimize

def get_correlation(data):
    return data.corr(method='pearson')

def calc_distance(correlation):
    distance_corr = np.sqrt(0.5 * (1 - correlation))
    return distance_corr

def euclidean_distance(distance_corr):
    # Calcula a distância euclidiana entre todas as combinações de pontos
    d_euclidean = (pdist(distance_corr, metric='euclidean'))
    return d_euclidean

def hierarchical_clustering(euclidean_distance, linkage):
    clustering_matrix = hr.linkage(euclidean_distance, method = linkage, optimal_ordering = True)
    return clustering_matrix

# algoritmo para obter a matriz de similariadade, atraves das alturas dos clusters
def construir_matriz_similaridade(clustering_matrix, assets):
    num_assets = len(assets)
    num_clusters = num_assets + len(clustering_matrix)
    
    # Inicializando a matriz de similaridade
    matriz_similaridade = np.zeros((num_clusters, num_clusters))

    # Mapeamento de clusters já formados
    cluster_map = {i: [i] for i in range(num_assets)}
    
    # Preenchendo a matriz de similaridade a partir da matriz de clusterização
    for i, (cluster1, cluster2, altura, _) in enumerate(clustering_matrix):
        cluster1, cluster2 = int(cluster1), int(cluster2)
        new_cluster = num_assets + i
        
        # Atualizando o mapa com o novo cluster formado
        cluster_map[new_cluster] = cluster_map[cluster1] + cluster_map[cluster2]
        
        # Atualizando a matriz de similaridade para todos os elementos dos clusters fundidos
        for elem1 in cluster_map[cluster1]:
            for elem2 in cluster_map[cluster2]:
                matriz_similaridade[elem1, elem2] = altura
                matriz_similaridade[elem2, elem1] = altura

    # Extraindo a submatriz correspondente apenas aos ativos originais
    final_matriz = matriz_similaridade[:num_assets, :num_assets]

    # Convertendo para DataFrame para melhor visualização
    matriz_df = pd.DataFrame(final_matriz, columns=assets, index=assets)
    return matriz_df

# função que transforma a matriz d_barra na matriz s_barra
def f(d_barra):
    '''
    The function f(.) can also be specified according to how D is defined.
    For example, suppose that D is defined such that d_ij = sqrt(k(1 - p_ij)).
    One could define s_ij using a quadradic function, as
        s_ij = 1 - (d_ij^2 / k)
    '''
    N = d_barra.shape[0]
    f_barra = (1 - ((d_barra ** 2) / 0.5)) / N
    return f_barra 

# algortimo que minimiza a função 19, porém como conversado na ultima reunião
# ao inves de maximizar deve ser performada a minimização sem levar em consideração
# o retorno esperado
def resolver_otimizacao(S_bar_df, gammas):
    S_bar = S_bar_df.values
    n = S_bar.shape[0]
    resultados = {}

    # Restrições
    restricoes = {'type': 'eq', 'fun': lambda b: np.sum(b) - 100}
    limites = [(0, 100) for _ in range(n)]
    b0 = np.ones(n) * (100 / n)  # chute inicial

    for gamma in gammas:
        gamma_val = 1.0 #if gamma == np.inf else gamma
        def objetivo(b):
            return - (gamma_val * b.T @ S_bar @ b) 

        res = minimize(objetivo, b0, method='SLSQP', bounds=limites, constraints=[restricoes])

        resultados[gamma] = res.x

    return resultados

# função que performa a equeção 17
def get_w_subi(b, data):
    w_i = []
    for i, j in enumerate(b):
        w_i.append(b[j] / data.std().values)
    return w_i

# função que performa a equação 18
def get_w_i_hrb(w_i):
    w_i_hrb = []
    for i, j in enumerate(w_i):
        w_i_hrb.append(w_i[i] / np.sum(w_i[i]))
    return w_i_hrb

def main(data, assets):
    correlation = get_correlation(data)                                     # Processo para obter matriz D
    distance_corr = calc_distance(correlation)                              # Processo para obter matriz D
    e_distance = euclidean_distance(distance_corr)                          # Aqui obtemos a matriz D
    clustering = hierarchical_clustering(e_distance, 'single')              # Aqui obtemos a matriz clustering para obter a matriz D_barra
    matriz_similaridade = construir_matriz_similaridade(clustering, assets) # Aqui obtemos a matriz D_barra

    s_barra = f(matriz_similaridade)                # Obtendo a matriz s_barra
    #gamma = [10, 20, 40, 80, np.inf]               # Definindo gamma como o autor
    gamma = [10]
    b = resolver_otimizacao(s_barra, gamma)         # Obtendo os valores de b após performar a minimização da equação 19
    w_i = get_w_subi(b, data)                       # Obtendo os valores de w_i após resolver equação 17
    w_i_hrb = get_w_i_hrb(w_i)                      # Obtendo os valores de w_i_hrb (budgets) após resolver equação 18

    budgets_portfolio = pd.DataFrame((w_i_hrb[i] for i in range(len(w_i_hrb))), index=gamma, columns=assets).T
    #budgets_portfolio.loc['soma'] = budgets_portfolio.sum()
    return budgets_portfolio[10].values