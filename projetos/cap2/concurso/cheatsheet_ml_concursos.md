# CHEAT SHEET: MACHINE LEARNING
## Guia Rápido para Concursos Públicos

---

## 📋 CICLO DE VIDA DO PROJETO ML

```
1. Definir Problema
2. Coletar Dados
3. Explorar Dados (EDA)
4. Dividir (Treino/Validação/Teste)
5. Pré-processar
6. Treinar Modelos
7. Ajustar Hiperparâmetros
8. Avaliar no Teste
9. Deploy
10. Monitorar
```

---

## 🔢 FÓRMULAS ESSENCIAIS

### Métricas de Regressão

```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²

RMSE = √MSE

MAE = (1/n) Σ|yᵢ - ŷᵢ|

R² = 1 - (SS_res / SS_tot)
   = 1 - [Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²]
```

### Feature Scaling

```
Padronização (StandardScaler):
z = (x - μ) / σ
Resultado: μ=0, σ=1

Normalização (MinMaxScaler):
x' = (x - min) / (max - min)
Resultado: x' ∈ [0,1]
```

### Correlação

```
Pearson: r = Cov(X,Y) / (σₓ × σᵧ)
Range: [-1, +1]
```

### Validação Cruzada

```
K-Fold: Score = (1/K) Σ Scoreᵢ
```

---

## 📊 TABELAS DE REFERÊNCIA RÁPIDA

### Quando Usar Cada Métrica (Regressão)

| Métrica | Sensível a Outliers | Interpretação | Uso Principal |
|---------|---------------------|---------------|---------------|
| **MAE** | Baixa | Erro médio absoluto | Com outliers |
| **MSE** | Alta | Erro quadrático | Otimização |
| **RMSE** | Alta | Mesma unidade que y | Comparação de modelos |
| **R²** | Média | % variância explicada | Qualidade geral |

### Modelos: Características Rápidas

| Modelo | Interpretável | Precisa Scaling | Não-Linear | Overfitting |
|--------|--------------|----------------|------------|-------------|
| **Linear Regression** | ✅ | ⚠️ | ❌ | Baixo |
| **Decision Tree** | ✅ | ❌ | ✅ | Alto |
| **Random Forest** | ⚠️ | ❌ | ✅ | Médio |
| **SVM** | ❌ | ✅ | ✅ | Médio |
| **KNN** | ⚠️ | ✅ | ✅ | Alto |

### Tratamento de Valores Ausentes

| Método | Tipo de Dado | Quando Usar |
|--------|--------------|-------------|
| **Média** | Numérico | Distribuição simétrica |
| **Mediana** | Numérico | Com outliers |
| **Moda** | Categórico | Categoria mais comum |
| **Constante** | Qualquer | Valor específico faz sentido |
| **Remover** | Qualquer | <5% missing, feature não crítica |

### Encoding de Variáveis Categóricas

| Método | Quando Usar | Exemplo |
|--------|-------------|---------|
| **Ordinal** | Ordem natural | Educação: [Fund, Médio, Superior] |
| **One-Hot** | Sem ordem | Cor: [Vermelho, Azul, Verde] |
| **Binary** | Apenas 2 categorias | Sexo: [M, F] → [0, 1] |
| **Frequency** | Alta cardinalidade | Codificar por frequência |

---

## ⚠️ REGRAS DE OURO

### 1. Ordem das Operações
```
✅ CORRETO:
1. Dividir dados
2. Fit no treino
3. Transform em treino e teste

❌ ERRADO:
1. Fit em todos os dados
2. Dividir
→ DATA LEAKAGE!
```

### 2. Fit vs Transform
```python
# ✅ CORRETO
scaler.fit(X_train)           # Aprende do treino
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ❌ ERRADO
scaler.fit(X_train)
scaler.fit(X_test)            # NÃO fit no teste!
```

### 3. Conjunto de Teste
```
✅ Use APENAS UMA VEZ no final
❌ Não use para ajustar modelo
❌ Não use para escolher features
❌ Não use para tunar hiperparâmetros
```

### 4. Validação Cruzada
```
✅ Apenas no conjunto de TREINO
❌ Não inclua dados de teste
```

---

## 🎯 DIAGNÓSTICOS RÁPIDOS

### Identificar Overfitting/Underfitting

```
           │  TREINO  │  TESTE   │  DIAGNÓSTICO
───────────┼──────────┼──────────┼─────────────────
Bom        │  Baixo   │  Baixo   │  ✅ Modelo OK
Overfitting│  Baixo   │  Alto    │  ❌ Reduzir complexidade
Underfitting│ Alto    │  Alto    │  ❌ Aumentar complexidade
```

### Soluções por Diagnóstico

**OVERFITTING (Treino << Teste)**:
- ➕ Mais dados de treino
- ➕ Regularização (L1, L2)
- ➖ Reduzir features
- ➖ Reduzir complexidade do modelo
- ✂️ Poda (árvores)
- 🎲 Dropout (redes neurais)
- 📊 Cross-validation

**UNDERFITTING (Treino ≈ Teste, ambos altos)**:
- ➕ Mais features (engenharia)
- ➕ Modelo mais complexo
- ➖ Reduzir regularização
- ⏱️ Treinar mais tempo

---

## 🔍 ANÁLISE EXPLORATÓRIA (EDA)

### Comandos Pandas Essenciais

```python
# Visão geral
df.head()                    # Primeiras linhas
df.info()                    # Tipos e nulos
df.describe()                # Estatísticas
df.shape                     # (linhas, colunas)

# Valores ausentes
df.isnull().sum()            # Contagem por coluna
df.isnull().mean() * 100     # Percentual

# Valores únicos
df['col'].nunique()          # Número de únicos
df['col'].value_counts()     # Frequência

# Correlação
df.corr()                    # Matriz
df.corr()['target'].sort_values()  # Com target
```

### Detectar Outliers (IQR)

```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['col'] < lower) | (df['col'] > upper)]
```

---

## 🛠️ CÓDIGO PADRÃO

### Pipeline Completo

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Definir colunas
num_cols = ['idade', 'salario']
cat_cols = ['cidade', 'sexo']

# Pipeline numérico
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline categórico
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

# Preprocessador
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# Pipeline completo
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Usar
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Divisão simples
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)

# Divisão estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # ou stratify=df['categoria']
    random_state=42
)
```

### Validação Cruzada

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-scores)
print(f"RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

### Salvar/Carregar Modelo

```python
import joblib

# Salvar
joblib.dump(model, 'model.pkl')
joblib.dump(pipeline, 'pipeline.pkl')

# Carregar
model = joblib.load('model.pkl')
pipeline = joblib.load('pipeline.pkl')

# Usar
predictions = model.predict(X_new)
```

---

## 📈 INTERPRETAÇÃO DE RESULTADOS

### Correlação de Pearson

```
r = +1.0  →  Correlação positiva perfeita
r = +0.7  →  Correlação positiva forte
r = +0.3  →  Correlação positiva fraca
r =  0.0  →  Sem correlação linear
r = -0.3  →  Correlação negativa fraca
r = -0.7  →  Correlação negativa forte
r = -1.0  →  Correlação negativa perfeita
```

### R² Score

```
R² = 1.0   →  Modelo perfeito
R² = 0.8   →  Explica 80% da variância (bom)
R² = 0.5   →  Explica 50% da variância (razoável)
R² = 0.0   →  Tão bom quanto prever a média
R² < 0.0   →  Pior que prever a média
```

### Feature Importance

```
Importância > 0.1  →  Feature muito importante
Importância ≈ 0.05 →  Feature relevante
Importância < 0.01 →  Considerar remover
```

---

## 🚨 ARMADILHAS COMUNS

### ❌ Erros Frequentes

1. **Normalizar ANTES de dividir**
```python
# ❌ ERRADO
X_scaled = scaler.fit_transform(X)
X_train, X_test = split(X_scaled)

# ✅ CORRETO
X_train, X_test = split(X)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

2. **Usar feature com info do futuro**
```python
# ❌ ERRADO: Prever inadimplência
features = ['renda', 'idade', 'dias_em_atraso']
# dias_em_atraso só existe APÓS inadimplência!

# ✅ CORRETO
features = ['renda', 'idade', 'historico_credito']
```

3. **Avaliar no teste múltiplas vezes**
```python
# ❌ ERRADO
model_v1.fit(X_train); test_v1 = score(X_test)
model_v2.fit(X_train); test_v2 = score(X_test)
model_v3.fit(X_train); test_v3 = score(X_test)
# Está ajustando ao teste!

# ✅ CORRETO
# Usar validação cruzada no treino
# Testar NO TESTE apenas uma vez no final
```

4. **Esquecer random_state**
```python
# ❌ Resultados não reproduzíveis
train_test_split(X, y, test_size=0.2)

# ✅ Reproduzível
train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🧠 CONCEITOS-CHAVE PARA DECORAR

### Tipos de Aprendizado

```
SUPERVISIONADO
├─ Classificação: Prever categoria
│  └─ Binária: 2 classes
│  └─ Multiclasse: 3+ classes
└─ Regressão: Prever valor numérico

NÃO-SUPERVISIONADO
├─ Clustering: Agrupar dados similares
└─ Redução de Dimensionalidade: PCA, t-SNE

SEMI-SUPERVISIONADO
└─ Poucos dados rotulados + muitos não-rotulados

POR REFORÇO
└─ Aprender por tentativa e erro (recompensas)
```

### Viés-Variância Trade-off

```
VIÉS ALTO (Underfitting)
→ Modelo muito simples
→ Não captura padrões
→ Erro alto em treino E teste

VARIÂNCIA ALTA (Overfitting)
→ Modelo muito complexo
→ Decora os dados
→ Erro baixo em treino, alto em teste

IDEAL
→ Equilíbrio entre viés e variância
→ Generaliza bem
```

### No Free Lunch Theorem

```
NÃO existe modelo melhor para TODOS os problemas
→ Sempre testar múltiplos algoritmos
→ Performance depende dos dados
→ Conhecimento do domínio é crucial
```

---

## 📚 GLOSSÁRIO ULTRA-RÁPIDO

| Termo | Definição |
|-------|-----------|
| **Feature** | Variável independente (coluna) |
| **Label/Target** | Variável dependente (y) |
| **Sample** | Linha, observação, instância |
| **Training** | Processo de aprendizado |
| **Inference** | Fazer predições |
| **Fit** | Aprender dos dados |
| **Transform** | Aplicar transformação aprendida |
| **Epoch** | Uma passada completa pelos dados |
| **Batch** | Subconjunto de dados |
| **Hyperparameter** | Parâmetro definido antes do treino |
| **Parameter** | Valor aprendido durante treino |
| **Baseline** | Modelo simples de referência |
| **Ensemble** | Combinar múltiplos modelos |
| **Pipeline** | Sequência automatizada de operações |
| **Leakage** | Vazamento de info do teste |
| **Curse of Dimensionality** | Problemas em alta dimensão |

---

## ✅ CHECKLIST PRÉ-PROVA

### Conhecimento Teórico
- [ ] Ciclo de vida do projeto ML
- [ ] Diferença treino/validação/teste
- [ ] Overfitting vs Underfitting
- [ ] Métricas de avaliação
- [ ] Validação cruzada
- [ ] Tipos de encoding
- [ ] Feature engineering
- [ ] Data leakage
- [ ] Viés-variância trade-off
- [ ] Modelos principais

### Habilidades Práticas
- [ ] Pandas (head, info, describe, corr)
- [ ] Train-test split
- [ ] Pipelines do sklearn
- [ ] Treinamento de modelos
- [ ] Grid/Random Search
- [ ] Cálculo de métricas
- [ ] Análise de erros
- [ ] Salvar/carregar modelos

### Fórmulas
- [ ] MSE, RMSE, MAE
- [ ] R²
- [ ] Padronização
- [ ] Normalização
- [ ] Correlação de Pearson

---

## 🎯 DICAS FINAIS PARA A PROVA

1. **Leia com atenção**: "conjunto de treino" ≠ "conjunto de teste"

2. **Ordem importa**: SEMPRE dividir ANTES de processar

3. **Fit = Aprender**: Apenas no treino!

4. **Transform = Aplicar**: Em treino E teste

5. **Teste é sagrado**: Use UMA VEZ no final

6. **Correlação ≠ Causalidade**

7. **Interpretação de métricas**:
   - RMSE alto? Modelo ruim ou target muito variável?
   - R² negativo? Modelo pior que média
   - CV com alta variância? Dados instáveis

8. **Outliers**: Mediana > Média

9. **Alta dimensão**: Nem sempre mais é melhor

10. **Desbalanceamento**: Use métricas adequadas (não acurácia!)

---

## 📖 RECURSOS ADICIONAIS

### Para Aprofundar
- Scikit-Learn Documentation
- Kaggle Learn
- Fast.ai
- Coursera ML courses
- Papers: arXiv.org

### Praticar
- Kaggle Competitions
- UCI ML Repository
- Google Dataset Search
- Projetos próprios

---

**BOA SORTE NA PROVA! 🍀**

*Este cheat sheet é um complemento ao Guia de Estudos completo.*

---

*Versão: 1.0*
*Atualizado: 2024*
