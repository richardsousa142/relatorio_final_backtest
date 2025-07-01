import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.linalg import solve, cholesky
from math import log
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist, squareform

'''
X-Means usando KMeans
'''

def get_correlation(data):
  return data.corr(method='pearson')

def calc_distance(correlation):
  distance_corr = np.sqrt(0.5 * (1 - correlation))
  return distance_corr

def euclidian_distance_improve(len_stocks, distance_df):
  eucli_dist = pd.DataFrame()
  for n in range(len_stocks):
    for k in range(( distance_df.shape[0] * distance_df.shape[1] )):
      row = int(np.floor(k / distance_df.shape[1]))
      col = k % distance_df.shape[1]
      eucli_dist.loc[row, col] = np.sqrt((distance_df.iloc[n, row] - distance_df.iloc[n, col]) ** 2)
  eucli_dist.index = distance_df.index
  eucli_dist.columns = distance_df.columns
  return eucli_dist

class XMeans:
    def __init__(self, k_min=2, k_max=10, max_iter=100):
        #self.data = data
        self.k_min = k_min
        self.k_max = k_max
        self.max_iter = max_iter

    def bic(self, x, centers, q, ignore_covar):
        lnl = self.likehood(x, centers, ignore_covar)
        bic = -2 * lnl[0] + q * np.log(x.shape[0])
        return bic

    def likehood(self, x, centers, ignore_covar):
        n = x.shape[0]; p = x.shape[1]
        if n <= 2: return np.nan, np.nan
        cov_x = np.cov(x, rowvar=False)
        if p == 1: 
            inversa = 1 / np.array(cov_x).flatten()
            determi = np.array(cov_x).flatten()
        else:
            if ignore_covar:
                inversa = np.diag(1 / np.diag(cov_x))
                determi = np.prod(np.diag(cov_x))
            else:
                inversa = np.linalg.inv(cov_x)   # inverse matrix of "cov_x"
                y = cholesky(cov_x, lower=False) # Cholesky decomposition
                determi = np.prod(np.diag(y))    # # vx = t(y) %*% y, where y is triangular, # then, det(vx) = det(t(y)) * det(y)
        t1 = -p/2 * 1.837877066 # 1.837... = log(2 * 3.1415...)
        t2 = -np.log(determi) / 2
        xmu = x - centers
        if p == 1:
            s = np.sum(xmu**2 * inversa)
        else:
            def tx_inv_vx_x(row, inversa):
                return np.dot(np.dot(row, inversa), row)
            s = sum(tx_inv_vx_x(row, inversa) for row in xmu)
        t3 = -s / 2
        ll = (t1 + t2) * n + t3.ravel() #log likelihood
        return ll, determi

    def bic_linha(self, x, kmeans, q, ignore_covar):
        #pegar cluster 0
        clj1 = x[kmeans.labels_ == 0]
        #pegar cluster 1
        clj2 = x[kmeans.labels_ == 1]

        #calcular likehood1
        lnl1, determi1 = self.likehood(clj1, kmeans.cluster_centers_[0], ignore_covar)
        #calcular likehood2
        lnl2, determi2 = self.likehood(clj2, kmeans.cluster_centers_[1], ignore_covar)

        #pegar n1 do cluster 0
        n1 = clj1.shape[0]
        #pegar n2 do cluster 1
        n2 = clj2.shape[0]

        #verificar determinante
        if np.isnan(determi1) or np.isnan(determi2):
            beta = 0
        else:
            #calcular beta
            beta = pdist(kmeans.cluster_centers_) / np.sqrt(determi1 + determi2)

        #calcular alfa
        alpha = 0.5 / stats.norm.cdf(beta)
        #calcular bic1
        bic1 = -2 * lnl1 + q * np.log(n1)
        #calcular bic2
        bic2 = -2 * lnl2 + q * np.log(n2)
        #calcular bic geral
        bic = -2 * lnl1 - 2 * lnl2 + 2 * q * np.log(n1 + n2) - 2 * (n1 + n2) * np.log(alpha)

        return {
            "bic": [bic1, bic2, bic],
            "lnl":[ lnl1, lnl2 ],
            "det_vx": [ determi1, determi2 ],
            "clj1": clj1,
            "clj2": clj2
            }

    def update_center(self, center, k1, k2, sub_centers):
        z = []
        for i in range(1, max(k2, 2)+1):
            if i == k1:
                z.append(sub_centers[0, :])
            elif i == k2:
                z.append(sub_centers[1, :])
            else:
                #print(center[i-1])
                z.append(center[i-1])
        return z

    def merge_result(self, kmeans, clsub, ik):
        cluster = kmeans.labels_
        centers = []
        size = []
        lnl = []
        determinante = []
        k = 0
        for i in range(ik):
            k += len(clsub[i][2])

        for i in range(ik, 0, -1):
            xsub = clsub[i-1][0]
            iki = ik -(i) +1
            centers.append(clsub[iki-1][1])
            size.append(clsub[iki-1][2])
                
            for j in range(len(clsub[i-1][2]), 0, -1):
                xsub[xsub == j] = k
                k = k - 1
            
            indices = np.where(cluster == (i-1))  
            cluster[indices] = xsub
        if k != 0: raise ValueError("mergeResult: assertion failed (k = 0)...")
        return {
            "cluster": cluster,
            "centers": centers,
            "lnL": lnl,
            "detVx": determinante,
            "size": size
            }

    def fit(self, data, seed, ignore_covar = True):
        # Passo 1 - prepare the p-dimensional data
        p = data.shape[1]
        q = 2 * p if ignore_covar else p * (p + 3) / 2
        clsub = []

        # Passo 1 - set initial number of clusters to be k0
        if self.k_min < 2: self.k_min = 2

        # Passo 2 - Apply KMeans to all data with k = k0. We name the
        # name the divided clusters as C1, C2, C3, ..., Ck0
        kmeans = KMeans(n_clusters=self.k_min, random_state=seed, max_iter=10, init='random', n_init=1).fit(data)
        clusters = kmeans.labels_

        # Passo 3 - Repeat the following procedure from step 4 to step 9
        # by setting i = 1, 2, 3, ..., k0
        for i in range(self.k_min):
            stack = []; k1 = 1; k2 = k1 + 1; flag = True

            # Passo 4 - For a cluster Ci apply k-means by setting k = 2.
            # We name the divided clusters as Ci_(1) e Ci_(2)
            sub_cluster = data[clusters == i]        # --> Cluster Ci | sub_cluster = yi
            sub_center  = kmeans.cluster_centers_[i] # --> Center do cluster Ci
            zi_center = sub_center
            yi_cluster = np.ones(len(data[clusters == i]) ,dtype=int)
            while True:
                if len(sub_cluster) == 1: break
                sub_kmeans = KMeans(n_clusters=2, random_state=seed, max_iter=10, init='random', n_init=1).fit(sub_cluster) # --> Ci_1 e Ci_2
                sub_clusters = sub_kmeans.labels_
                sub_centerss = sub_kmeans.cluster_centers_

                # Passo 5 - We assume the following p-dimensional normal distribution
                # for the data xi contained in Ci
                if flag:
                    bic = self.bic(sub_cluster, sub_center, q, ignore_covar)
                    flag = False
                
                # Passo 6 - We assume the following p-dimensional normal distribution
                # with their param θi_1 e θi_2 for cluster Ci_1 e Ci_2 respectively
                bic_linha = self.bic_linha(sub_cluster, sub_kmeans, q, ignore_covar)
                
                # Passo 7 - if BIC > BIC' the two divided model is preferred and the
                # division continues; we set Ci <- Ci_1. And stacked Ci_2 
                if bic > bic_linha['bic'][2]:
                    sub_clusters = np.where(sub_clusters == 1, k2, sub_clusters)  # Substitui 1 por k2
                    sub_clusters = np.where(sub_clusters == 0, k1, sub_clusters)  # Substitui 0 por k1
                    #Substituir valores em v com base em k1
                    for i in range(len(yi_cluster)):
                        if yi_cluster[i] == 1:
                            yi_cluster[i] = sub_clusters[i % len(sub_clusters)]

                    zi_center  = self.update_center(zi_center, k1, k2, sub_centerss)
                    clj2 = sub_cluster[sub_kmeans.labels_ == 1]
                    clj2_center = sub_centerss[1]; lnl2 = bic_linha['lnl'][1]; bic2 = bic_linha['bic'][1]
                    stack.append(  (clj2, clj2_center, lnl2, bic2, k2)  )

                # Passo 8 - if BIC < BIC' clusters are not longer divided. Extract 
                # the stacked data which is stored in step 7 and set Ci <- Ci_2
                # and return to step 4, if the stack is empty go to step 9
                if bic < bic_linha['bic'][2] or np.isnan(bic_linha['bic'][2]):
                    if stack:
                        elementos = stack.pop()
                        sub_cluster = elementos[0]
                        sub_center = elementos[1]
                        lnl = elementos[2]
                        bic = elementos[3]
                        k1 = elementos[4]
                        k2 = k2
                        continue
                    break
                
                sub_cluster = sub_cluster[sub_kmeans.labels_ == 0]
                sub_kmeans.cluster_centers_ = sub_centerss[0]  
                
                # Passo 9 - The 2-division procedure for Ci is completed. We renumber the
                # cluster identification such that it becomes unique in Ci
                k2 += 1
                # end of while
                
            size = np.bincount(yi_cluster, minlength=yi_cluster.max() + 1)[1:]
            clsub.append(  (yi_cluster, zi_center, size)  )

        #end of for  
        # Passo 10 - The 2-division procedure for initial k0 divided clusters is
        # completed. We renumber all clusters identifications such that they 
        # become unique
        xcl = self.merge_result(kmeans, clsub, self.k_min)

        # Passo 11 - Output the cluster identification number to which 
        # each element is allocated, the center of each cluster, the log
        # likehood of each cluster, and the number of elements in each 
        # cluster 
        return {
                "cluster": xcl["cluster"],
                "centers": xcl["centers"],
                "lnl": xcl["lnL"],
                "size": xcl["size"]
            }

    # Função para calcular o risco total do portfólio
    def portfolio_risk(self, weights, covariance_matrix):
        return np.sqrt(weights.T @ covariance_matrix @ weights)

    # Função para calcular o MRC
    def marginal_risk_contribution(self, weights, covariance_matrix, portfolio_risk):
        return (covariance_matrix @ weights) / portfolio_risk

    # Função objetivo para minimizar diferenças nas contribuições de risco
    def clustering_risk_parity_objective(self, weights, covariance_matrix, clusters, k, Nk):
        portfolio_risk_ = self.portfolio_risk(weights, covariance_matrix)
        mrc = self.marginal_risk_contribution(weights, covariance_matrix, portfolio_risk_)
        trc = np.sum(weights * mrc)
        rc = weights * mrc / trc  # Contribuição de risco normalizada
        # calcular a funcao objetivo
        penalty = 0
        for i, label in enumerate(clusters):
            target = (1 / k) * (1 / Nk[label])
            penalty += (rc[i] - target)**2
        return penalty

    def peso(self, cluster, cov):
        """ Pesos do portfólio (Clustering Risk Parity) """
        np.random.seed(42)
        # Dados do clustering
        clusters = cluster  # Resultado do clustering
        unique_clusters = np.unique(clusters)

        # Número de clusters e ativos por cluster
        unique_labels, counts = np.unique(clusters, return_counts=True)
        k = len(unique_labels)
        Nk = dict(zip(unique_labels, counts))
        n_assets = len(clusters)
        
        # Restrições
        initial_weights = np.ones(n_assets) / n_assets  # Pesos iniciais iguais
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Soma dos pesos = 1
        )
        bounds = [(0, 1) for _ in range(n_assets)] 
        # Matriz de covariância
        covariance_matrix = cov
        # Otimização
        result = minimize(
            self.clustering_risk_parity_objective,
            initial_weights,
            args=(covariance_matrix, clusters, k, Nk),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            #options={'disp':True}
        )

        # Pesos finais
        optimized_weights = result.x
        return optimized_weights

def main(data, cov, asset, seed):
    correlation = get_correlation(data)
    distance_corr = calc_distance(correlation)
    distance_euclidean = squareform(pdist(distance_corr, metric='euclidean'))
    X_train = distance_euclidean
    xm = XMeans()
    xms = xm.fit(X_train, seed)
    w = xm.peso(xms['cluster'], cov)
    return w
 
 