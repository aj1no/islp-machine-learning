import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Carregando o Dataset
# Vamos usar um dataset real muito famoso em Machine Learning: "Breast Cancer" (Câncer de Mama).
# O objetivo é prever se o tumor é maligno (0) ou benigno (1).
print("Carregando o dataset de Câncer de Mama...")
data = load_breast_cancer()
X = data.data    # As características do tumor (raio, textura, área, etc)
y = data.target  # Os diagnósticos reais (0 ou 1)
nomes_classes = data.target_names # 'malignant' e 'benign'

# 2. Dividindo os dados
# Separamos 80% dos dados para treinar o modelo e deixamos 20% separados só para testá-lo.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Criando e Treinando o Modelo (Random Forest)
print("Treinando o modelo de Machine Learning (Random Forest)...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train) # O modelo está "aprendendo" os padrões

# 4. Fazendo Predições
# Agora vamos fazer com que o modelo tente prever os diagnósticos usando os 20% do teste
print("Fazendo predições nos dados de teste...")
y_pred = modelo.predict(X_test)

# 5. Avaliando os Resultados (Matriz de Confusão)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Relatório de Classificação Detalhado ---")
print(classification_report(y_test, y_pred, target_names=nomes_classes))

print("--- Matriz de Confusão (Texto) ---")
print(cm)
# Linha 1: Casos que eram malignos
# Linha 2: Casos que eram benignos
print("-" * 35)

# 6. Plotando Graficamente a Matriz de Confusão
plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f"Predição: {n}" for n in nomes_classes], 
            yticklabels=[f"Real: {n}" for n in nomes_classes])

plt.title('Matriz de Confusão - Diagnóstico de Câncer de Mama')
plt.ylabel('Valores Reais (O que o dado diz)')
plt.xlabel('Valores Preditos (O que o modelo achou)')

# Mostrando o gráfico na tela!
plt.show()

# ===========================================================================
# PARTE 2: Exercícios de Regressão Logística (ISLP)
# ===========================================================================

import pandas as pd
from ISLP import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def ajustar_regressao_logistica_py(nome_dataset, coluna_alvo, p_teste=0.3, semente=42):
    print(f"\nTentando carregar o dataset {nome_dataset}...")
    try:
        # Carregar o dataset usando a biblioteca ISLP
        df = load_data(nome_dataset)
    except Exception as e:
        print(f"Atenção: Houve um erro ao importar '{nome_dataset}' do ISLP. Verifique se a biblioteca está instalada.")
        return None
    
    # Separar a variável dependente (y) das numéricas/nominais (X)
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]
    
    # O scikit-learn precisa de números nas variáveis independentes (transformando texto em bool/int)
    X = pd.get_dummies(X, drop_first=True)
    
    # Particionamento: usamos o train_test_split que já foi importado no início do arquivo
    X_trein, X_test, y_trein, y_test = train_test_split(
        X, y, test_size=p_teste, random_state=semente, stratify=y
    )
    
    # Modelo
    modelo_lr = LogisticRegression(max_iter=10000)
    modelo_lr.fit(X_trein, y_trein)
    
    # Predição e Avaliação (confusion_matrix já importada)
    predicoes = modelo_lr.predict(X_test)
    acuracia = accuracy_score(y_test, predicoes)
    matriz_conf = confusion_matrix(y_test, predicoes)
    
    # Imprimir Relatório
    print("=======================================================")
    print(f"Dataset: ISLP::{nome_dataset} | Variável Alvo: '{coluna_alvo}'")
    print(f"N (Treino): {len(X_trein)} | N (Teste): {len(X_test)}")
    print("\nMatriz de Confusão:")
    
    rotulos = y.unique()
    matriz_df = pd.DataFrame(matriz_conf, index=[f"Real {r}" for r in rotulos], columns=[f"Pred {r}" for r in rotulos])
    print(matriz_df)
    print(f"\nAcurácia Teste: {acuracia:.4f}")
    print("=======================================================\n")
    
    return modelo_lr

# Execução
print("\n--- a) Regressão Logística no Dataset DEFAULT ---")
modelo_default = ajustar_regressao_logistica_py('Default', 'student')

print("--- b) Regressão Logística no Dataset SMARKET ---")
modelo_smarket = ajustar_regressao_logistica_py('Smarket', 'Direction')

print("--- c) Regressão Logística no Dataset WEEKLY ---")
modelo_weekly = ajustar_regressao_logistica_py('Weekly', 'Direction')

print("--- d) Regressão Logística no Dataset CARAVAN ---")
modelo_caravan = ajustar_regressao_logistica_py('Caravan', 'Purchase')

# ===========================================================================
# PARTE 3: Regressão Logística Básica (Sem separação Treino/Teste)
# ===========================================================================
def ajustar_regressao_basica_py(nome_dataset, coluna_alvo):
    print(f"\nTentando carregar o dataset {nome_dataset} (SEM PARTICIONAMENTO)...")
    try:
        df = load_data(nome_dataset)
    except Exception as e:
        print(f"Erro ao importar '{nome_dataset}' do ISLP.")
        return None
    
    # Separar dependente (y) e independentes (X)
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]
    
    # Dummy para qualitativas do scikit-learn
    X = pd.get_dummies(X, drop_first=True)
    
    # 1. Ajuste/Treino COM TODOS OS DADOS (sem train_test_split)
    modelo_lr = LogisticRegression(max_iter=10000)
    modelo_lr.fit(X, y)
    
    # 2. Predição sendo testada na mesma base que aprendeu (Otimista)
    predicoes = modelo_lr.predict(X)
    
    # 3. Avaliação de Resultados
    acuracia = accuracy_score(y, predicoes)
    matriz_conf = confusion_matrix(y, predicoes)
    
    print("=======================================================")
    print(f"Dataset: ISLP::{nome_dataset} | Variável Alvo: '{coluna_alvo}'")
    print(f"N (Total de dados para Treino e Teste): {len(X)}")
    print("\nMatriz de Confusão:")
    
    rotulos = y.unique()
    matriz_df = pd.DataFrame(matriz_conf, index=[f"Real {r}" for r in rotulos], columns=[f"Pred {r}" for r in rotulos])
    print(matriz_df)
    print(f"\nAcurácia na mesma base de treino: {acuracia:.4f}")
    print("=======================================================\n")
    
    return modelo_lr

# --- Execução da Parte 3 ---
print("\n--- a) Regressão BÁSICA no Dataset DEFAULT ---")
mod_basico_default = ajustar_regressao_basica_py('Default', 'student')

print("--- b) Regressão BÁSICA no Dataset SMARKET ---")
mod_basico_smarket = ajustar_regressao_basica_py('Smarket', 'Direction')

print("--- c) Regressão BÁSICA no Dataset WEEKLY ---")
mod_basico_weekly = ajustar_regressao_basica_py('Weekly', 'Direction')

print("--- d) Regressão BÁSICA no Dataset CARAVAN ---")
mod_basico_caravan = ajustar_regressao_basica_py('Caravan', 'Purchase')
