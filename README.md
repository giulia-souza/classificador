# Classificador de triagem de vítimas

Projeto desenvolvido na disciplina de Sistemas Inteligentes para estudar a
classificação de vítimas de acordo com a prioridade de atendimento. O projeto
gera dados sintéticos de sinais clínicos e compara dois modelos de
aprendizado supervisionado:

- **Árvore de decisão CART**;
- **Rede neural MLP**.

As classes utilizadas seguem a classificação de triagem:

| Código | Classe |
| ---: | --- |
| 0 | Verde |
| 1 | Amarelo |
| 2 | Vermelho |
| 3 | Preto |

## Requisitos

- Python 3.9 ou superior;
- `numpy`;
- `pandas`;
- `scikit-learn`;
- `matplotlib`;
- `joblib`.

Instale as dependências com:

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

## Estrutura do projeto

```text
.
├── classificador.py          # Treina e avalia CART e rede neural com validação cruzada
├── dataset_vitimas.py        # Gera o dataset sintético de vítimas
├── teste_cego.py             # Retreina os melhores modelos e executa o teste cego
├── datasets/
│   └── vict/
│       └── 1300v/
│           ├── data_treino.csv
│           └── data_testeCego.csv
├── melhor_cart.joblib        # Modelo CART salvo
├── melhor_rn.joblib          # Modelo de rede neural salvo
├── cart_matriz_2.png         # Matriz de confusão do CART
├── rn_matriz_2.png           # Matriz de confusão da rede neural
└── Relatorio Classificador - SI.pdf
```

## Dados

O dataset possui variáveis clínicas e características derivadas:

`idade`, `fc`, `fr`, `pas`, `spo2`, `temp`, `pr`, `sg`, `fx`, `queim`,
`gcs`, `avpu`, `tri` e `sobr`.

Durante o treinamento, as colunas `gcs`, `avpu`, `tri` e `sobr` são removidas
das entradas. A variável `tri` é o alvo de classificação; portanto, o modelo
utiliza os sinais e características clínicas restantes para prever a classe de
triagem.

Os arquivos de dados usados pelos scripts ficam em
`datasets/vict/1300v/`. O gerador cria `data.csv`; para usá-lo diretamente no
treinamento, salve ou copie uma versão com o nome esperado pelo script
(`data_treino.csv`).

## Como executar

### 1. Gerar um dataset sintético

```bash
python dataset_vitimas.py
```

O comando gera 10.000 vítimas, distribuídas inicialmente entre as quatro
classes, aplica ruído controlado e salva o resultado em:

```text
datasets/vict/1300v/data.csv
```

O script também exibe a distribuição das classes e um histograma da
probabilidade de sobrevivência.

### 2. Treinar e comparar os modelos

Com `data_treino.csv` disponível no diretório esperado, execute:

```bash
python classificador.py
```

O treinamento usa validação cruzada estratificada com cinco divisões e
`F1-macro` como métrica de seleção. Para a árvore, são comparados diferentes
valores de `max_depth`. Para a rede neural, são comparadas arquiteturas com
uma, cinco e três camadas ocultas de 300 neurônios.

O script imprime, para cada configuração, as médias e os desvios-padrão das
métricas de treino e validação.

### 3. Executar o teste cego

Com `data_testeCego.csv` disponível, execute:

```bash
python teste_cego.py
```

Esse script:

1. carrega o dataset de teste cego;
2. retreina os modelos selecionados com todos os dados disponíveis;
3. salva os modelos em `melhor_cart.joblib` e `melhor_rn.joblib`;
4. calcula precisão, recall, F1-macro e acurácia;
5. exibe os relatórios de classificação e as matrizes de confusão.

As janelas das matrizes de confusão são abertas pelo `matplotlib`; feche-as
para concluir a execução.

## Reprodutibilidade

As etapas principais usam sementes aleatórias fixas (`random_state` e `seed`)
para tornar os resultados reproduzíveis. Alterações no nível de ruído, na
distribuição das classes ou nos parâmetros dos modelos podem mudar os
resultados.

## Observações

- Os dados são sintéticos e não devem ser usados para decisões médicas reais.
- Os arquivos `.joblib` representam modelos treinados a partir dos dados
  presentes neste repositório.
- O projeto não contém um serviço de inferência ou uma interface gráfica; a
  execução é feita pelos scripts Python.
- O projeto foi realizado para a matéria Sistemas Inteligentes ministrada
pelo professor Cesar Augusto Tacla - UTFPR CT
