# Estratégia de Backtest com Rolling Window e X-Means Clustering

Este projeto implementa uma estratégia de backtest baseada em uma abordagem de **agrupamento (clustering)** para alocação de ativos via **Clustering Risk Parity**, utilizando uma variação do algoritmo **X-Means**.

---

## ⚙️ Estrutura Geral da Estratégia

A lógica central da estratégia é baseada em um **rolling window**: uma janela de tempo é usada para estimar os parâmetros do portfólio (**janela in-sample**), e o retorno é avaliado no **período seguinte (out-of-sample)**.

---

## 🔁 Rolling Window: In-Sample vs Out-of-Sample

### Parâmetros:
- `InS`: Tamanho da janela **in-sample** (quantidade de períodos usada para estimar os pesos do portfólio).
- `OoS`: Quantidade total de períodos que serão avaliados **fora da amostra** (horizonte do backtest).

### Processo:

Para cada passo da janela:
1. **Seleciona as datas da janela in-sample.**
2. **Filtra os ativos mais líquidos** (composição) e **remove ativos com dados faltantes (NaNs)**.
3. **Obtém os retornos dos ativos in-sample.**
4. **Calcula o retorno out-of-sample** dos mesmos ativos no mês seguinte.
5. **Executa o algoritmo X-Means Clustering** com diferentes sementes para formar os clusters e aplicar a **Clustering Risk Parity**, gerando os **pesos otimizados**.
6. **Armazena os resultados**:
   - Retorno do portfólio no mês seguinte.
   - Peso dos ativos.
   - Índice de concentração do portfólio (soma dos quadrados dos pesos).
   - Turnover, medindo a rotatividade do portfólio (mudança nos pesos entre períodos consecutivos).

---

## 🧠 Clustering Risk Parity com X-Means

A lógica de alocação é baseada na ideia de **paridade de risco por clusters**, onde:

- Os ativos são agrupados via **X-Means**, que determina automaticamente o número ideal de clusters com base no critério **BIC**.
- Os pesos são calculados de forma a **equalizar as contribuições marginais de risco** dos ativos dentro de cada cluster.
- Essa abordagem visa diversificar melhor o risco entre grupos de ativos similares.

---

## 📈 Métricas Calculadas

Durante o backtest, as seguintes métricas são computadas para avaliar a qualidade da estratégia de alocação:

- **📊 Retorno do Portfólio (Out-of-Sample)**  
  Mede o desempenho da alocação em um período fora da amostra (OoS), ou seja, o retorno obtido com os pesos calculados no mês anterior aplicados aos retornos reais do mês seguinte.

- **📉 Concentração do Portfólio**  
  Avalia o nível de diversificação do portfólio. Uma concentração alta indica que poucos ativos estão recebendo grande parte do capital. É calculada como a soma dos quadrados dos pesos:  
  `Concentracao = soma(w_i ** 2)`  
  Quanto menor esse valor, mais diversificado está o portfólio.

- **🔁 Turnover**  
  Mede a rotatividade dos ativos no portfólio, ou seja, o quanto os pesos mudam de um período para o outro. Um turnover alto implica em maior custo de transação. É calculado com base na diferença entre os pesos ajustados pelos retornos entre os períodos consecutivos:
  ```python
  calculate_to(w_t_1, w_t, r_oos_full, capital)
