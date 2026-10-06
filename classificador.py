import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

## ----------------------------------- preparando os dados
BASE_FOLDER = Path("./datasets/vict/1300v")
BASE_FOLDER.mkdir(parents=True, exist_ok=True)

df_dataset = pd.read_csv(f'{BASE_FOLDER}/data_treino.csv')

colunas_proibidas = ['gcs', 'avpu', 'tri', 'sobr']
X = df_dataset.drop(columns=colunas_proibidas)
y = df_dataset['tri']

print(f"Tamanho total do dataset (Treino/Validação): {len(X)} vítimas\n")

## ----------------------------------- arvore de decisao (CART)

validCruzada = StratifiedKFold(n_splits=5, shuffle=True, random_state=46)#divide em 5 partes - 4 para treino e 1 para validação e faz 5 rodadas para validacao cruzada

modelo_cart = DecisionTreeClassifier()

parametros = {'max_depth': [3, 8, None]}#arvore 3(subajustado), 8(equilibrado), None(sobreajustado)

grid = GridSearchCV(modelo_cart, parametros, cv=validCruzada, scoring='f1_macro', return_train_score=True)

grid.fit(X, y)

resultados = pd.DataFrame(grid.cv_results_)

modelos_nomes = ['U (max_depth=3)', 'E (max_depth=8)', 'O (max_depth=None)']

for i, nome in enumerate(modelos_nomes):
    print(f"\nMODELO {nome}:")
    
    f1_treino_folds = np.array([resultados.loc[i, f'split{k}_train_score'] for k in range(5)]) # f1_macro de treino em cada fold
    f1_val_folds = np.array([resultados.loc[i, f'split{k}_test_score'] for k in range(5)]) # f1_macro de validação em cada fold

    diferencas_abs = np.abs(f1_treino_folds - f1_val_folds)
    
    # calcula medias
    media_treino = np.mean(f1_treino_folds)
    media_val = np.mean(f1_val_folds)
    media_diff = np.mean(diferencas_abs)
    
    # calcula desvPad amostral (DPA)
    #ddof = 1 pra amostral, ddof = 0 pra populacional
    dpa_treino = np.std(f1_treino_folds, ddof=1)
    dpa_val = np.std(f1_val_folds, ddof=1)
    dpa_diff = np.std(diferencas_abs, ddof=1)
    
    print(f"TREINO f1 médio (DPA)     | {media_treino:.5f} ({dpa_treino:.5f})")
    print(f"VALIDAÇÃO f1 médio (DPA)  | {media_val:.5f} ({dpa_val:.5f})")
    print(f"MÉDIA DAS DIFS. (DPA)     | {media_diff:.5f} ({dpa_diff:.5f})")

    
## ----------------------------------- rede neural
scaler = StandardScaler() 
X_norm = scaler.fit_transform(X)#precisa ser normalizado p funcionar usando o sgd!!

modelo_rn = MLPClassifier(max_iter=3000, alpha=0.0, random_state=42)

parametros_rn = {
    'hidden_layer_sizes': [(1,), (5,), (300, 300, 300)], #(1) para U, (5) para E, (300,300,300) para O
    'activation': ['relu'],
    'solver': ['sgd'],
    'learning_rate_init': [0.03] #indicado pelo prof
}

grid_rn = GridSearchCV(modelo_rn, parametros_rn, cv=validCruzada, scoring='f1_macro', return_train_score=True)

print("Treinando as Redes Neurais...")
grid_rn.fit(X_norm, y)

resultados_rn = pd.DataFrame(grid_rn.cv_results_)
modelos_nomes_rn = ['U (1)', 'E (5)', 'O (300, 300, 300)']

for i, nome in enumerate(modelos_nomes_rn):
    print(f"\nMODELO {nome}:")
    
    f1_treino_folds = np.array([resultados_rn.loc[i, f'split{k}_train_score'] for k in range(5)])
    f1_val_folds = np.array([resultados_rn.loc[i, f'split{k}_test_score'] for k in range(5)])
    
    diferencas_abs = np.abs(f1_treino_folds - f1_val_folds)
    
    media_treino = np.mean(f1_treino_folds)
    media_val = np.mean(f1_val_folds)
    media_diff = np.mean(diferencas_abs)
    
    dpa_treino = np.std(f1_treino_folds, ddof=1)
    dpa_val = np.std(f1_val_folds, ddof=1)
    dpa_diff = np.std(diferencas_abs, ddof=1)
    
    print(f"TREINO f1 médio (DPA)     | {media_treino:.5f} ({dpa_treino:.5f})")
    print(f"VALIDAÇÃO f1 médio (DPA)  | {media_val:.5f} ({dpa_val:.5f})")
    print(f"MÉDIA DAS DIFS. (DPA)     | {media_diff:.5f} ({dpa_diff:.5f})")

