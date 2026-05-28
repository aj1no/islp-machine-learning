# Exercícios de Machine Learning - ISLP

*[Read in English](README.md)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557c?style=flat-square)](https://matplotlib.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/aj1no/islp-machine-learning/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/aj1no/islp-machine-learning/actions)

Este repositório contém exercícios resolvidos de classificação usando modelos de **Machine Learning** no Python com `scikit-learn` e o pacote didático `ISLP`. O código-fonte explora diferentes algoritmos de classificação e os efeitos na precisão estatística ao separar os dados de Treino/Teste vs usar a base completa.

---

## O que tem no código?

O script principal `exercicios_logistica.py` conta com 3 partes práticas:

1. **Random Forest e Dataset de Câncer de Mama:** Carrega os dados nativos do `sklearn`, treina o modelo separando 80%/20%, plota o heatmap de uma matriz de confusão e gera um relatório de classificação completo.
2. **Regressão Logística com Separação de Dados:** Re-executa 4 exercícios clássicos da biblioteca `ISLP`, dividindo a modelagem com amostras independentes e validando em partes não conhecidas da tabela usando a notação matricial de binarização (`get_dummies`).
   - `ISLP::Default` (Alvo: *student*)
   - `ISLP::Smarket` (Alvo: *Direction*)
   - `ISLP::Weekly` (Alvo: *Direction*)
   - `ISLP::Caravan` (Alvo: *Purchase*)
3. **Regressão Logística Básica:** Remove o `train_test_split` e treina/verifica o modelo na base de dados inteira, demonstrando o conceito estatístico de subestimação do erro de treinamento (avaliar o modelo com dados conhecidos).

---

## Bibliotecas e Instalação

Para poder rodar o mesmo projeto, tenha o Python 3.10+ e execute o comando abaixo no seu terminal para obter os requisitos fundamentais:

```bash
pip install ISLP pandas scikit-learn matplotlib seaborn
```

Feito isso, execute o arquivo principal:
```bash
python exercicios_logistica.py
```
*(Tenha em mente que as Regressões da Parte 2 e 3 só mostrarão seus relatórios no terminal logo após você fechar a janela da figura do Matplotlib que abrirá em tela na Parte 1).*
