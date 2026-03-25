# 📊 ISLP Machine Learning Exercises

Este repositório contém exercícios resolvidos de classificação usando modelos de **Machine Learning** no Python com `scikit-learn` e o pacote didático `ISLP`. O código-fonte explora diferentes algoritmos de classificação e os efeitos na precisão estatística ao separar os dados de Treino/Teste vs usar a base completa.

## 🚀 O que tem no código?

O script principal `exercicios_logistica.py` conta com 3 partes práticas:

1. **Random Forest e Dataset de Câncer de Mama**: Carrega os dados nativos do `sklearn`, treina o modelo separando 80%/20%, plota o heatmap de uma matriz de confusão e solta um relatório completo.
2. **Regressão Logística com Separação de Dados**: Refaz 4 exercícios clássicos da biblioteca `ISLP`, dividindo a modelagem com amostras independentes e validando em partes virgens da tabela usando a notação matricial de binarização (`get_dummies`).
   - `ISLP::Default` (Alvo: *student*)
   - `ISLP::Smarket` (Alvo: *Direction*)
   - `ISLP::Weekly` (Alvo: *Direction*)
   - `ISLP::Caravan` (Alvo: *Purchase*)
3. **Regressão Logística Básica**: Remove o `train_test_split` e treina e verifica as informações na base de forma inteira com a demonstração estatística do modelo prever amostras que ele próprio já conhecia ("Erro de Treinamento Superestimado").

## 🛠 Bibliotecas e Instalação
Para poder rodar o mesmo projeto, tenha o Python 3.10+ e rode o comando abaixo no seu terminal para puxar os requisitos fundamentais:

```bash
pip install ISLP pandas scikit-learn matplotlib seaborn
```

Feito isso, é só plugar o arquivo e rodar!
```bash
python exercicios_logistica.py
```
*(Tenha em mente que as Regressões da Parte 2 e 3 só mostrarão seus relatórios no terminal logo após você fechar a janela da figura do Matplotlib que abrirá em tela na Parte 1).*
