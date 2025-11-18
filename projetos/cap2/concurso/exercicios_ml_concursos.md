# EXERCÍCIOS PRÁTICOS E QUESTÕES ADICIONAIS
## Complemento ao Guia de Estudos de Machine Learning

---

## PARTE 1: EXERCÍCIOS DE CÓDIGO

### Exercício 1: Análise Exploratória de Dados

**Contexto**: Você recebeu um dataset de vendas com as seguintes colunas:
- `data_venda` (datetime)
- `valor_venda` (float)
- `regiao` (categoria)
- `produto` (categoria)
- `quantidade` (int)

**Tarefa**: Escreva código Python para:

a) Identificar valores ausentes em cada coluna
```python
# Sua resposta:
```

b) Calcular estatísticas descritivas apenas para colunas numéricas
```python
# Sua resposta:
```

c) Encontrar a correlação entre `valor_venda` e `quantidade`
```python
# Sua resposta:
```

d) Identificar e contar outliers em `valor_venda` usando o método IQR
```python
# Sua resposta:
```

**GABARITO**:

```python
# a) Identificar valores ausentes
missing_values = df.isnull().sum()
print(missing_values)

# Ou mais detalhado:
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Percentage': missing_percentage
})
print(missing_df)

# b) Estatísticas descritivas
numeric_stats = df.describe()
# Ou especificando:
numeric_stats = df[['valor_venda', 'quantidade']].describe()

# c) Correlação
correlation = df['valor_venda'].corr(df['quantidade'])
print(f"Correlação: {correlation:.3f}")

# d) Outliers usando IQR
Q1 = df['valor_venda'].quantile(0.25)
Q3 = df['valor_venda'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['valor_venda'] < lower_bound) | 
              (df['valor_venda'] > upper_bound)]
print(f"Número de outliers: {len(outliers)}")
```

---

### Exercício 2: Pré-processamento de Dados

**Dataset**:
```
   idade  salario  cidade      comprou
0   25    NaN      SP          Sim
1   30    5000     RJ          Não
2   35    6000     SP          Sim
3   NaN   4500     MG          Não
4   28    NaN      SP          Sim
```

**Tarefa**: Complete o pipeline de pré-processamento:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Features numéricas e categóricas
numeric_features = ['idade', 'salario']
categorical_features = ['cidade']

# Complete o código:
numeric_pipeline = Pipeline([
    ('imputer', ____________________________),  # Usar mediana
    ('scaler', _____________________________)   # Padronizar
])

categorical_pipeline = Pipeline([
    ('imputer', ____________________________),  # Usar moda
    ('encoder', _____________________________)  # OneHot
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])
```

**GABARITO**:

```python
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Usar:
X_transformed = preprocessor.fit_transform(X_train)
```

---

### Exercício 3: Detecção de Data Leakage

**Identifique o erro em cada código**:

**Código A**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

X_train, X_test = train_test_split(df_scaled, test_size=0.2)
```

**Código B**:
```python
# Remover linhas duplicadas
df_clean = df.drop_duplicates()

# Dividir
X_train, X_test = train_test_split(df_clean, test_size=0.2)
```

**Código C**:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

imputer = SimpleImputer()
imputer.fit(X)  # Fit em todos os dados
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
```

**GABARITO**:

**Código A - ERRO**: Data leakage
- Problema: `fit_transform` foi aplicado ANTES da divisão
- Solução: Dividir primeiro, depois aplicar scaler

**Código B - SEM ERRO**: 
- Remover duplicatas antes de dividir é correto
- Não causa data leakage

**Código C - ERRO**: Data leakage
- Problema: `imputer.fit(X)` usa TODO o dataset
- Solução: `imputer.fit(X_train)` apenas

---

### Exercício 4: Validação Cruzada

**Complete o código**:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

# Realizar validação cruzada com 5 folds
# Usar RMSE como métrica
scores = cross_val_score(
    _____________,  # modelo
    _____________,  # features
    _____________,  # target
    cv=___________,  # número de folds
    scoring='______'  # métrica (neg_mean_squared_error)
)

# Converter para RMSE
rmse_scores = ___________________________

# Exibir resultados
print(f"RMSE médio: {_________.mean():.2f}")
print(f"Desvio padrão: {_________.std():.2f}")
```

**GABARITO**:

```python
scores = cross_val_score(
    model,  
    X_train,  
    y_train,  
    cv=5,  
    scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-scores)

print(f"RMSE médio: {rmse_scores.mean():.2f}")
print(f"Desvio padrão: {rmse_scores.std():.2f}")

# Análise adicional:
print(f"Melhor fold: {rmse_scores.min():.2f}")
print(f"Pior fold: {rmse_scores.max():.2f}")
print(f"Amplitude: {rmse_scores.max() - rmse_scores.min():.2f}")
```

---

### Exercício 5: Grid Search

**Complete o Grid Search**:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Definir grade de hiperparâmetros
param_grid = {
    'n_estimators': _____________,  # [50, 100, 200]
    'max_depth': ______________,     # [10, 20, None]
    'min_samples_split': _______     # [2, 5, 10]
}

# Total de combinações: ___________

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=_____,  # 5 folds
    scoring='_________________',  # neg_mean_squared_error
    n_jobs=-1
)

grid_search.fit(_________, _________)

# Exibir melhores parâmetros
print("Melhores parâmetros:", _______________)
print("Melhor score:", ________________)
```

**GABARITO**:

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Total de combinações: 3 × 3 × 3 = 27

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor score:", np.sqrt(-grid_search.best_score_))

# Análise adicional:
results_df = pd.DataFrame(grid_search.cv_results_)
top5 = results_df.nsmallest(5, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score']
]
print("\nTop 5 configurações:")
print(top5)
```

---

## PARTE 2: QUESTÕES DE MÚLTIPLA ESCOLHA AVANÇADAS

### Bloco 1: Pré-processamento

**Q1**: Considere um dataset com a feature "temperatura" variando de -10°C a 40°C. Após aplicar StandardScaler, os valores transformados estarão aproximadamente em qual faixa?

a) [0, 1]  
b) [-10, 40]  
c) [-3, 3]  
d) [-1, 1]  

**Resposta**: c) [-3, 3]
**Justificativa**: StandardScaler centraliza em 0 e padroniza para std=1. Pela regra empírica, ~99.7% dos dados ficam entre [-3, 3] (3 desvios padrão).

---

**Q2**: Em um dataset com 1000 linhas, uma feature tem 200 valores ausentes. Você decide remover todas as linhas com valores ausentes nesta feature. Qual a principal consequência?

a) Melhor acurácia do modelo  
b) Redução de 20% do dataset, possível perda de informação valiosa  
c) Aumento do overfitting  
d) Impossibilidade de treinar o modelo  

**Resposta**: b) Redução de 20% do dataset, possível perda de informação valiosa

---

**Q3**: Qual afirmação sobre One-Hot Encoding é INCORRETA?

a) Cria N-1 colunas binárias para N categorias (com drop='first')  
b) É adequado para variáveis ordinais  
c) Pode causar curse of dimensionality com muitas categorias  
d) Evita o modelo interpretar ordem inexistente entre categorias  

**Resposta**: b) É adequado para variáveis ordinais
**Justificativa**: Para variáveis ordinais, use OrdinalEncoder para preservar a ordem.

---

### Bloco 2: Modelagem

**Q4**: Um modelo de Random Forest com n_estimators=100 demora 10 minutos para treinar. Aproximadamente quanto tempo levará com n_estimators=200 e n_jobs=1 (um core)?

a) 10 minutos (mesmo tempo)  
b) 15 minutos  
c) 20 minutos  
d) 40 minutos  

**Resposta**: c) 20 minutos
**Justificativa**: Tempo é aproximadamente linear com número de árvores. 2x árvores ≈ 2x tempo.

---

**Q5**: Qual modelo NÃO pode fornecer probabilidades de predição para classificação?

a) Logistic Regression  
b) Random Forest  
c) K-Nearest Neighbors  
d) Support Vector Machine (sem probability=True)  

**Resposta**: d) Support Vector Machine (sem probability=True)
**Justificativa**: SVM padrão fornece apenas decisões binárias. Precisa de `probability=True` para probabilidades (com custo computacional adicional).

---

**Q6**: Em um problema de regressão, você observa:
- Training RMSE: 1000
- Validation RMSE: 1500
- Test RMSE: 1520

Qual ação é MAIS apropriada?

a) Aumentar complexidade do modelo  
b) Coletar mais dados de treinamento  
c) Remover features  
d) Usar apenas o conjunto de teste daqui em diante  

**Resposta**: b) Coletar mais dados de treinamento
**Justificativa**: Gap entre treino (1000) e validação/teste (1500) indica overfitting. Mais dados ajudam a reduzir overfitting.

---

### Bloco 3: Avaliação

**Q7**: Para um modelo de regressão, você calcula:
- MAE = 10.000
- RMSE = 15.000

O que isso indica?

a) Erro de cálculo, pois RMSE deve ser menor que MAE  
b) Presença de alguns erros grandes (outliers nas predições)  
c) Modelo com performance ruim  
d) Dataset com muitos valores ausentes  

**Resposta**: b) Presença de alguns erros grandes (outliers nas predições)
**Justificativa**: RMSE > MAE indica que o quadrado dos erros aumentou significativamente, sugerindo erros grandes.

---

**Q8**: Um modelo tem R² = -0.5. O que isso significa?

a) Erro de cálculo impossível  
b) Modelo pior que simplesmente prever a média  
c) Correlação negativa perfeita  
d) Modelo com 50% de acurácia  

**Resposta**: b) Modelo pior que simplesmente prever a média
**Justificativa**: R² < 0 significa que o modelo é pior que o baseline (prever sempre a média).

---

**Q9**: Em validação cruzada 5-fold, você obteve RMSEs: [100, 150, 95, 300, 110]. O que isso sugere?

a) Modelo consistente e confiável  
b) Possível outlier ou distribuição não uniforme dos dados  
c) Underfitting severo  
d) Necessidade de aumentar para 10 folds  

**Resposta**: b) Possível outlier ou distribuição não uniforme dos dados
**Justificativa**: O fold 4 (300) está muito diferente dos outros, sugerindo dados problemáticos ou não-representativos naquele fold.

---

### Bloco 4: Conceitos Avançados

**Q10**: Qual técnica NÃO ajuda a combater overfitting?

a) Regularização L1 (Lasso)  
b) Dropout  
c) Feature Engineering (adicionar mais features)  
d) Early Stopping  

**Resposta**: c) Feature Engineering (adicionar mais features)
**Justificativa**: Adicionar features AUMENTA a complexidade e pode piorar overfitting. As outras técnicas reduzem complexidade/regularizam.

---

**Q11**: Em um Grid Search com 20 combinações de hiperparâmetros e CV=10, quantos modelos são treinados?

a) 20  
b) 30  
c) 200  
d) 2000  

**Resposta**: c) 200
**Cálculo**: 20 combinações × 10 folds = 200 treinamentos

---

**Q12**: Qual afirmação sobre validação cruzada é VERDADEIRA?

a) Deve ser usada no conjunto de teste final  
b) Sempre resulta em melhor performance que holdout  
c) Fornece estimativa mais robusta da performance do modelo  
d) Deve usar diferentes modelos em cada fold  

**Resposta**: c) Fornece estimativa mais robusta da performance do modelo

---

## PARTE 3: QUESTÕES DISSERTATIVAS

### Questão 1: Pipeline Completo

**Enunciado**: Descreva, passo a passo, o pipeline completo de um projeto de Machine Learning desde a coleta de dados até o deployment, destacando as boas práticas em cada etapa.

**Resposta Modelo**:

**1. Definição do Problema (Business Understanding)**:
- Entender objetivo de negócio
- Definir métrica de sucesso
- Estimar recursos necessários
- Identificar stakeholders

**2. Coleta de Dados**:
- Identificar fontes de dados
- Automatizar coleta (ETL)
- Validar qualidade inicial
- Documentar proveniência

**3. Exploração e Visualização (EDA)**:
- Análise estatística descritiva
- Identificar padrões e anomalias
- Visualizar distribuições
- Análise de correlações
- Detectar outliers e valores ausentes

**4. Divisão dos Dados**:
- **BOA PRÁTICA**: Dividir ANTES de qualquer processamento
- Proporção típica: 80% treino, 20% teste
- Usar estratificação se necessário
- Separar conjunto de validação ou usar CV

**5. Pré-processamento**:
- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Feature scaling/normalization
- **BOA PRÁTICA**: Fit apenas no treino, transform em treino e teste

**6. Feature Engineering**:
- Criar features derivadas
- Transformações (log, sqrt, etc.)
- Seleção de features
- Análise de importância

**7. Seleção e Treinamento de Modelos**:
- Testar múltiplos algoritmos
- Começar com modelos simples (baseline)
- Usar validação cruzada
- **BOA PRÁTICA**: Treinar apenas com dados de treino

**8. Ajuste Fino (Fine-tuning)**:
- Grid Search ou Random Search
- Otimizar hiperparâmetros
- Analisar feature importance
- Ensemble methods se necessário

**9. Avaliação Final**:
- Testar no conjunto de teste (UMA VEZ)
- Calcular múltiplas métricas
- Análise de erros
- Intervalo de confiança
- **BOA PRÁTICA**: Documentar todos os resultados

**10. Deployment**:
- Serializar modelo e pipeline
- Criar API ou interface
- Documentar uso
- Configurar monitoramento

**11. Monitoramento e Manutenção**:
- Monitorar performance
- Detectar data drift
- Retreinar periodicamente
- Versionamento de modelos
- A/B testing se necessário

---

### Questão 2: Comparação de Algoritmos

**Enunciado**: Compare Random Forest e Regressão Linear em termos de: (a) interpretabilidade, (b) sensibilidade a outliers, (c) necessidade de feature scaling, (d) capacidade de capturar relações não-lineares, (e) propensão a overfitting.

**Resposta Modelo**:

| Aspecto | Regressão Linear | Random Forest |
|---------|-----------------|---------------|
| **Interpretabilidade** | **Alta**: Coeficientes mostram impacto direto de cada feature | **Média/Baixa**: Feature importance, mas relações complexas difíceis de explicar |
| **Sensibilidade a Outliers** | **Alta**: Outliers afetam significativamente os coeficientes | **Baixa**: Árvores são robustas a outliers individuais |
| **Feature Scaling** | **Importante**: Melhora convergência (gradiente descente) e interpretação | **Desnecessário**: Baseado em divisões, não em distâncias |
| **Relações Não-lineares** | **Limitado**: Apenas relações lineares. Precisa de engenharia manual (x², interações) | **Alto**: Captura automaticamente interações complexas e não-linearidades |
| **Overfitting** | **Baixo**: Modelo simples, high bias/low variance | **Alto** (se não regularizado): Árvores profundas podem decorar dados |

**Quando usar cada um**:

**Regressão Linear**:
- Dados com relação predominantemente linear
- Interpretabilidade é crítica
- Dataset pequeno
- Baseline inicial

**Random Forest**:
- Relações complexas e não-lineares
- Features com escalas muito diferentes
- Presença de outliers
- Alta dimensionalidade
- Performance é mais importante que interpretabilidade

---

### Questão 3: Data Leakage

**Enunciado**: Explique o conceito de data leakage, forneça 3 exemplos práticos de como pode ocorrer, e descreva estratégias para preveni-lo.

**Resposta Modelo**:

**Definição**: Data leakage ocorre quando informação do conjunto de teste (ou do futuro, em produção) "vaza" para o treinamento, resultando em performance artificialmente alta que não se replica em dados novos.

**Exemplo 1: Target Leakage - Feature derivada do target**
```
Problema: Prever inadimplência de empréstimo
Feature suspeita: "dias_em_atraso"

❌ ERRO: Usar "dias_em_atraso" como feature
- Esta informação só existe APÓS a inadimplência
- Na predição real, não teremos este valor

✅ CORRETO: Usar apenas informação disponível no momento da aprovação
- Histórico de crédito
- Renda
- Razão dívida/renda
```

**Exemplo 2: Train-Test Contamination - Processar antes de dividir**
```python
❌ ERRO:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Usa TODO o dataset
X_train, X_test = train_test_split(X_scaled)

✅ CORRETO:
X_train, X_test = train_test_split(X)  # Divide PRIMEIRO
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Exemplo 3: Temporal Leakage - Usar dados futuros**
```
Problema: Prever vendas de janeiro usando dados de 2023
Feature suspeita: "média_vendas_2023"

❌ ERRO: A "média_vendas_2023" inclui vendas de janeiro, fevereiro, etc.
- Estamos usando informação do futuro

✅ CORRETO: Usar apenas dados históricos
- "média_vendas_2022"
- "média_vendas_últimos_90_dias" (terminando em dezembro)
```

**Estratégias de Prevenção**:

1. **Dividir Primeiro, Processar Depois**:
   - SEMPRE dividir dados antes de qualquer transformação
   - Fit apenas no treino, transform em treino e teste

2. **Usar Pipelines**:
   ```python
   pipeline = Pipeline([
       ('imputer', SimpleImputer()),
       ('scaler', StandardScaler()),
       ('model', RandomForestRegressor())
   ])
   # Pipeline garante que fit ocorre apenas no treino
   ```

3. **Validação Temporal** (para séries temporais):
   - Treinar com dados passados
   - Testar com dados futuros
   - Nunca usar dados futuros no treino

4. **Análise Crítica de Features**:
   - Perguntar: "Esta informação estará disponível na hora da predição?"
   - Desconfiar de features com correlação muito alta (>0.95) com target

5. **Cross-Validation Adequada**:
   - Para séries temporais: TimeSeriesSplit
   - Nunca embaralhar dados temporais

6. **Monitoramento em Produção**:
   - Se performance real << performance em teste → investigar leakage

---

### Questão 4: Curse of Dimensionality

**Enunciado**: Explique o fenômeno "Curse of Dimensionality". Como ele afeta algoritmos de Machine Learning? Quais técnicas podem mitigar seus efeitos?

**Resposta Modelo**:

**Definição**:
Curse of Dimensionality refere-se aos problemas que surgem quando trabalhamos com dados de alta dimensionalidade (muitas features). À medida que o número de dimensões aumenta, o espaço de features cresce exponencialmente, tornando os dados cada vez mais esparsos.

**Fenômenos Principais**:

1. **Esparsidade dos Dados**:
   - Em alta dimensão, pontos ficam muito distantes
   - Densidade de pontos decresce exponencialmente
   - Exemplo: 
     * 1D: 10 pontos em [0,1] → densidade 10/1 = 10
     * 2D: 10 pontos em [0,1]×[0,1] → densidade 10/1 = 10
     * 10D: 10 pontos em [0,1]^10 → densidade 10/1 ≈ 0

2. **Distâncias Tornam-se Similares**:
   - Em alta dimensão, todas as distâncias ficam parecidas
   - Conceito de "vizinho mais próximo" perde significado
   - Afeta algoritmos como KNN drasticamente

3. **Aumento do Overfitting**:
   - Mais features = mais complexidade = mais chance de decorar
   - Modelo pode encontrar padrões espúrios

**Impacto em Algoritmos**:

| Algoritmo | Impacto | Explicação |
|-----------|---------|------------|
| **KNN** | Severo | Baseado em distância; todas ficam similares |
| **SVM** | Alto | Espaço de alta dimensão dificulta encontrar hiperplano |
| **Regressão Linear** | Médio | Multicolinearidade e overfitting |
| **Random Forest** | Baixo | Feature selection implícita em cada split |
| **Redes Neurais** | Alto (sem regularização) | Muitos parâmetros podem decorar |

**Exemplo Numérico**:

Considere classificar imagens 100×100 pixels (10.000 dimensions):
- Para cobrir 0.1% do espaço com densidade de 1 ponto por unidade
- Precisaríamos de 10^10000 pontos!
- Isso é mais que o número de átomos no universo (10^80)

**Técnicas de Mitigação**:

**1. Feature Selection**:
```python
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=20)
X_selected = selector.fit_transform(X_train, y_train)
```
- Remove features irrelevantes ou redundantes
- Métodos: Correlação, feature importance, testes estatísticos

**2. Dimensionality Reduction - PCA**:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Manter 95% da variância
X_reduced = pca.fit_transform(X_train)
```
- Transforma features em componentes principais não-correlacionados
- Perde interpretabilidade mas mantém informação

**3. Regularização**:
```python
from sklearn.linear_model import Lasso, Ridge

# L1 (Lasso): Force alguns coeficientes a zero
model_l1 = Lasso(alpha=1.0)

# L2 (Ridge): Penaliza coeficientes grandes
model_l2 = Ridge(alpha=1.0)
```
- L1: Feature selection automática (sparse)
- L2: Reduz magnitude dos pesos

**4. Engenharia de Features**:
- Criar features mais informativas
- Combinar múltiplas features em uma
- Reduzir dimensões manualmente com conhecimento do domínio

**5. Usar Modelos Adequados**:
- Random Forest: Robusto a alta dimensionalidade
- Gradient Boosting: Lida bem com muitas features
- Evitar: KNN, métodos baseados em distância sem redução

**Exemplo Prático**:

Problema: Classificar textos com Bag-of-Words (50.000 palavras únicas)

**Abordagem Ruim**:
```python
# 50.000 features!
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
model = KNeighborsClassifier()  # Sofrerá muito
model.fit(X, y)
```

**Abordagem Melhor**:
```python
# Reduzir dimensionalidade
vectorizer = CountVectorizer(max_features=1000)  # Top 1000 palavras
X = vectorizer.fit_transform(documents)

# Ou usar TF-IDF + SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

# Usar modelo robusto
model = RandomForestClassifier()
model.fit(X_reduced, y)
```

---

## PARTE 4: ESTUDOS DE CASO

### Caso 1: Previsão de Churn de Clientes

**Contexto**: Uma empresa de telecomunicações quer prever quais clientes vão cancelar o serviço (churn) no próximo mês.

**Dataset**:
- 10.000 clientes
- Features: tempo_cliente (meses), gasto_mensal, chamadas_suporte, idade, tipo_plano
- Target: churn (Sim/Não) - 25% taxa de churn

**Questões**:

1. **Este é um problema de classificação ou regressão? Binário ou multiclasse?**

**Resposta**: Classificação binária (duas classes: Churn Sim/Não)

2. **Qual métrica seria MAIS apropriada para este problema e por quê?**

**Resposta**: 
- **Recall (Sensibilidade)** ou **F1-Score**
- Justificativa: 
  * Custo de perder cliente (falso negativo) > custo de oferta preventiva (falso positivo)
  * Acurácia pode ser enganosa com classes desbalanceadas (75% não-churn)
  * Queremos identificar o máximo de potenciais churners

3. **O dataset está desbalanceado. Quais técnicas você usaria?**

**Resposta**:
- **Amostragem Estratificada**: Manter 25% churn em treino e teste
- **SMOTE**: Criar amostras sintéticas da classe minoritária
- **Pesos de Classe**: `class_weight='balanced'` no modelo
- **Ensemble com Balanceamento**: BalancedRandomForest

4. **Um consultor sugere incluir a feature "dias_desde_ultima_reclamação". Isso é uma boa ideia?**

**Resposta**: 
- **Depende!** 
- Se "reclamação" pode ocorrer APÓS a decisão de churn → **Target Leakage**
- Se "reclamação" sempre precede o churn → **OK, pode usar**
- Melhor: Usar apenas informações definitivamente anteriores

---

### Caso 2: Previsão de Preços de Imóveis

**Contexto**: Prever preço de casas em uma cidade.

**Features Disponíveis**:
- area_m2, num_quartos, num_banheiros, idade_imovel
- latitude, longitude, distancia_centro
- valor_iptu, taxa_condominio
- data_venda

**Problema Encontrado**:
Seu modelo tem:
- R² treino: 0.95
- R² teste: 0.55

**Questões**:

1. **Qual é o problema principal?**

**Resposta**: **Overfitting severo**
- Grande gap entre treino (0.95) e teste (0.55)
- Modelo decorou os dados de treino

2. **Liste 5 ações concretas para melhorar a generalização.**

**Resposta**:

a) **Regularização**:
```python
# Ridge Regression
from sklearn.linear_model import Ridge
model = Ridge(alpha=10.0)

# Ou Lasso
from sklearn.linear_model import Lasso
model = Lasso(alpha=1.0)
```

b) **Reduzir Complexidade do Modelo**:
```python
# Se usando Random Forest
model = RandomForestRegressor(
    max_depth=10,  # Limitar profundidade
    min_samples_leaf=20  # Mínimo por folha
)
```

c) **Feature Selection**:
```python
# Remover features menos importantes
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(RandomForestRegressor(), threshold='median')
X_selected = selector.fit_transform(X_train, y_train)
```

d) **Coletar Mais Dados de Treino**:
- Mais amostras reduzem overfitting
- Especialmente efetivo se dataset é pequeno

e) **Validação Cruzada Durante Desenvolvimento**:
```python
scores = cross_val_score(model, X_train, y_train, cv=5)
# Ajustar hiperparâmetros baseado em CV, não em treino
```

3. **A feature "data_venda" deve ser usada? Como?**

**Resposta**:
- **NÃO usar data diretamente**
- **Extrair informações temporais**:
  ```python
  df['ano'] = df['data_venda'].dt.year
  df['mes'] = df['data_venda'].dt.month
  df['trimestre'] = df['data_venda'].dt.quarter
  df['dias_desde_2020'] = (df['data_venda'] - pd.Timestamp('2020-01-01')).dt.days
  ```
- **Considerar sazonalidade**: Preços podem variar por mês/trimestre

---

### Caso 3: Sistema de Recomendação

**Contexto**: Recomendar produtos para usuários em e-commerce.

**Dados**:
- Histórico de compras
- Avaliações (1-5 estrelas)
- Comportamento de navegação
- Dados demográficos

**Desafio**: Novo usuário sem histórico (cold start problem)

**Questões**:

1. **Que tipo de abordagem de ML usaria para usuários COM histórico?**

**Resposta**: **Collaborative Filtering**
- Baseado em usuários similares
- Matrix Factorization (SVD, ALS)
- Deep Learning (Autoencoders)

2. **E para usuários SEM histórico (cold start)?**

**Resposta**: **Content-Based Filtering**
- Usar dados demográficos
- Produtos populares (baseline)
- Dados de navegação inicial
- Transferir padrões de usuários similares

3. **Como avaliar a qualidade das recomendações?**

**Resposta**:

**Métricas Offline**:
- **Precision@K**: Dos K produtos recomendados, quantos o usuário gostou?
- **Recall@K**: Dos produtos que o usuário gostou, quantos foram recomendados?
- **NDCG**: Considera ordem da recomendação
- **AUC-ROC**: Para relevância binária

**Métricas Online (A/B Testing)**:
- Taxa de clique (CTR)
- Taxa de conversão
- Receita gerada
- Engajamento do usuário

---

## PARTE 5: ARMADILHAS COMUNS (PEGADINHAS)

### Pegadinha 1: Normalização

**Questão**: Você aplicou MinMaxScaler e os valores ficaram entre [0, 1]. Verdadeiro ou Falso: "Novos dados de teste ficarão SEMPRE entre [0, 1] também."

**Resposta**: **FALSO**

**Explicação**:
```python
# Treino: valores entre [10, 100]
scaler = MinMaxScaler()
scaler.fit(X_train)  # Aprende min=10, max=100

# Teste: contém valor 120 (fora do range do treino!)
X_test_scaled = scaler.transform(X_test)
# Valor 120 → (120-10)/(100-10) = 110/90 = 1.22 (FORA de [0,1])
```

**Lição**: MinMaxScaler usa min/max do TREINO. Dados de teste podem ter valores fora do range.

---

### Pegadinha 2: Validação Cruzada

**Questão**: Com CV=5, você treinou um modelo. Qual modelo deve ser usado em produção?

a) O modelo do fold com melhor performance  
b) A média dos 5 modelos  
c) Retreinar com TODO o dataset de treino  
d) Usar o último fold  

**Resposta**: **c) Retreinar com TODO o dataset de treino**

**Explicação**: CV é para AVALIAÇÃO, não para escolher modelo final. Depois de decidir hiperparâmetros com CV, retreine com todos os dados de treino.

---

### Pegadinha 3: Feature Importance

**Questão**: Random Forest mostra que "idade" tem importância 0.001. Verdadeiro ou Falso: "Podemos remover 'idade' sem afetar o modelo."

**Resposta**: **FALSO (com ressalvas)**

**Explicação**:
- Importância baixa pode indicar:
  1. Feature realmente irrelevante ✓
  2. Feature redundante (correlacionada com outra mais importante)
  3. Feature importante apenas em interação com outras
  
**Teste correto**:
```python
# Treinar SEM a feature
model_without = RandomForestRegressor()
model_without.fit(X_train_sem_idade, y_train)

# Treinar COM a feature
model_with = RandomForestRegressor()
model_with.fit(X_train_com_idade, y_train)

# Comparar performance
# Se diferença < threshold → pode remover
```

---

### Pegadinha 4: Correlação

**Questão**: Correlação entre X e Y é 0.02. Verdadeiro ou Falso: "X não tem relação com Y."

**Resposta**: **FALSO**

**Explicação**:
- Correlação de Pearson mede apenas relação **LINEAR**
- Pode haver relação não-linear forte
- Exemplo: Y = X²
  ```
  X:  -2  -1   0   1   2
  Y:   4   1   0   1   4
  Correlação ≈ 0, mas há relação óbvia!
  ```

---

### Pegadinha 5: Amostragem

**Questão**: Dataset tem 10.000 linhas. Você seleciona linhas 0-7999 para treino e 8000-9999 para teste. Qual o problema?

**Resposta**: **Não é aleatório!**

**Problema**:
- Se dados têm ordem temporal → treino e teste de períodos diferentes
- Se dados ordenados por categoria → treino e teste com distribuições diferentes
- Viés de seleção

**Solução**: SEMPRE embaralhar antes de dividir (exceto séries temporais!)

---

## CONCLUSÃO DO CADERNO DE EXERCÍCIOS

Este caderno complementa o Guia de Estudos com:
- ✅ Exercícios práticos de código
- ✅ Questões de múltipla escolha avançadas
- ✅ Questões dissertativas detalhadas
- ✅ Estudos de caso realistas
- ✅ Pegadinhas comuns em provas

**Próximos Passos**:
1. Resolver todos os exercícios sem olhar gabaritos
2. Implementar os casos práticos em um notebook Jupyter
3. Criar seu próprio projeto end-to-end
4. Revisar conceitos com dificuldade
5. Fazer simulados cronometrados

**Tempo Estimado**: 20-30 horas adicionais

Bons estudos! 🚀

---

*Versão: 1.0*
*Complemento ao: Guia de Estudos ML para Concursos*
