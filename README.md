# Desafio Dio -  Criando Modelos com Python e Machine Learning para Prever a Evolução do COVID-19 no Brasil



A pandemia de COVID-19 teve um impacto devastador no Brasil, tornando crucial o desenvolvimento de ferramentas para prever sua evolução e orientar medidas de mitigação eficazes. Este projeto abrangente visa construir modelos de aprendizado de máquina usando Python para prever casos, mortes e recuperações do COVID-19 no Brasil, fornecendo insights valiosos para formuladores de políticas e autoridades de saúde.



### **Metodologia:**



#### **1. Coleta e Preparação de Dados:**

- **Coleta de dados:** Coletar dados de fontes confiáveis, como o Ministério da Saúde, a Universidade Johns Hopkins e o Google Cloud BigQuery.
- **Pré-processamento:** Limpar, pré-processar e transformar os dados para garantir consistência e qualidade. Isso inclui lidar com valores ausentes, normalizar dados e converter dados categóricos em numéricos.
- **Divisão de dados:** Dividir os dados em conjuntos de treinamento, validação e teste, garantindo que o conjunto de treinamento seja representativo da população real.



#### **2. Exploração e Análise de Dados:**

- **Análise exploratória:** Explorar os dados usando visualizações (como gráficos e tabelas) e estatísticas descritivas para identificar padrões, tendências e outliers.

  

- **Análise de correlação:** Analisar as correlações entre diferentes variáveis, como casos, mortes, recuperações e fatores como medidas de distanciamento social e vacinação.

  

- **Identificação de recursos:** Identificar os recursos mais influentes que afetam a evolução da pandemia, usando técnicas como seleção de recursos e importância de recursos.



#### **3. Desenvolvimento e Avaliação do Modelo:**

- **Seleção do modelo:** Selecionar algoritmos de aprendizado de máquina adequados, como Regressão Linear, Árvores de Decisão, Random Forests e Redes Neurais.

  

- **Treinamento do modelo:** Treinar os modelos usando os dados de treinamento e avaliar seu desempenho usando os dados de validação.

  

- **Otimização do modelo:** Otimizar os modelos ajustando hiperparâmetros (por exemplo, taxa de aprendizado, número de árvores) e técnicas de regularização (por exemplo, L1, L2).

  

- **Avaliação do modelo:** Avaliar o desempenho dos modelos usando métricas como precisão, revocação, pontuação F1 e perda quadrada média (MSE).



#### **4. Previsão e Insights:**

- **Previsão:** Usar os modelos treinados para prever a evolução do COVID-19 no Brasil, incluindo casos, mortes e recuperações.

  

- **Análise de previsões:** Analisar as previsões para identificar tendências, pontos de inflexão potenciais e cenários de "e se".

  

- **Insights:** Fornecer insights e recomendações com base nas previsões do modelo, orientando medidas de saúde pública, políticas governamentais e estratégias de mitigação.



#### **5. Implantação e Monitoramento:**



- **Implantação do modelo:** Implantar os modelos treinados em uma plataforma de produção para fazer previsões em tempo real.

  

- **Monitoramento do modelo:** Monitorar o desempenho dos modelos implantados e fazer ajustes conforme necessário para garantir precisão e confiabilidade contínuas.



#### **Código de Amostra:**

python

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
dados = pd.read_csv('dados_covid19_brasil.csv')

# Pré-processar os dados
dados = dados.dropna()
dados['data'] = pd.to_datetime(dados['data'])

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(dados[['casos', 'mortes', 'recuperados']], dados['data'], test_size=0.2)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Avaliar o modelo
previsoes = modelo.predict(X_test)
mse = mean_squared_error(y_test, previsoes)
print("MSE:", mse)

# Prever os casos futuros
novos_dados = {'casos': [1000], 'mortes': [100], 'recuperados': [500], 'data': ['2023-03-01']}
novos_dados = pd.DataFrame(novos_dados)
previsao_futura = modelo.predict(novos_dados)
print("Previsão futura:", previsao_futura)
```



## Observação



A partir do projeto original e do acesso a dados mais atualizados crie uma comparação entre o que foi previsto através do método ARIMA utilizando as biblioteca pmdarima e o que de fato aconteceu. 

Da mesma forma criei um comparativo entre o previsto através da biblioteca fbprophet e os dados reais. Neste caso criei uma comparação com uma previsão de um milhão de infectados, dez milhões de infectatos e os mais de duzentos e onze milhões de brasileiros infectados.

A grande diferença do trabalho original foi a necessidade de fazer um agrupamento nos dados uma vez que a partir de uma determinada data o Brasil passou a ter dados por estados.



## **Conclusão:**

Este projeto abrangente visa fornecer modelos de aprendizado de máquina robustos e confiáveis para prever a evolução do COVID-19 no Brasil. As previsões e insights gerados pelo projeto podem auxiliar na tomada de decisões informadas, permitindo que as autoridades de saúde e os formuladores de políticas respondam de forma eficaz à pandemia e mitiguem seu impacto sobre a população brasileira.



### ARIMA

A conclusão que podemos chegar é que o previsto no ARIMA foi conservador uma vez que o número de infectados nos quinze dias posteriores foi maior que o método havia previsto.



### Prophet

Já o observado na comparação com o Prophet é que as medidas tomadas desaceleraram a curva de contagem comparando com a previsão feita usando esta biblioteca.
