import pandas as pd
from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler

BASE_FOLDER = Path("./datasets/vict/1300v")
BASE_FOLDER.mkdir(parents=True, exist_ok=True)

df_dataset = pd.read_csv(f'{BASE_FOLDER}/data_testeCego.csv')

print("\nIniciando retreino com a totalidade das 1300 vítimas (sem divisão)...")
X = df_dataset.drop(columns=['gcs', 'avpu', 'tri', 'sobr'])
y = df_dataset['tri']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

melhor_cart = DecisionTreeClassifier(max_depth=8, min_samples_leaf=1, criterion='gini', random_state=42)
melhor_rn = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', solver='sgd', learning_rate_init=0.03, max_iter=3000, random_state=42)

# Retreino
melhor_cart.fit(X, y)
melhor_rn.fit(X_norm, y)

joblib.dump(melhor_cart, 'melhor_cart.joblib')
joblib.dump(melhor_rn, 'melhor_rn.joblib')
print("'melhor_cart.joblib', 'melhor_rn.joblib' gerados com sucesso")

# Preparar os dados
df_cego = pd.read_csv(f'{BASE_FOLDER}/data_testeCego.csv')
X_cego = df_cego.drop(columns=['gcs', 'avpu', 'tri', 'sobr'])
y_cego = df_cego['tri']


X_cego_norm = scaler.transform(X_cego) 

y_pred_cart = melhor_cart.predict(X_cego)
y_pred_rn = melhor_rn.predict(X_cego_norm)

def calcular_metricas(y_true, y_pred, nome_modelo):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n--- {nome_modelo} ---")
    print(f"MÉDIA PRECISÃO (macro) : {prec:.5f}")
    print(f"MÉDIA RECALL (macro)   : {rec:.5f}")
    print(f"F1 SCORE (macro)       : {f1:.5f}")
    print(f"ACURÁCIA               : {acc:.5f}")

calcular_metricas(y_cego, y_pred_cart, "Melhor CART (E)")
calcular_metricas(y_cego, y_pred_rn, "Melhor RN (E)")

print("\nÁrvore de Decisão (CART) - Teste Cego")
print(classification_report(y_cego, y_pred_cart, zero_division=0))

disp_cart = ConfusionMatrixDisplay.from_predictions(
    y_cego, 
    y_pred_cart, 
    display_labels=['Verde (0)', 'Amarelo (1)', 'Vermelho (2)', 'Preto (3)'],
    cmap='viridis',
    normalize='true'
)
disp_cart.ax_.set_title("Matriz de Confusão - Teste Cego (CART)")
plt.show()

print("\nRede Neural (RN) - Teste Cego")
print(classification_report(y_cego, y_pred_rn, zero_division=0))

disp_rn = ConfusionMatrixDisplay.from_predictions(
    y_cego, 
    y_pred_rn, 
    display_labels=['Verde (0)', 'Amarelo (1)', 'Vermelho (2)', 'Preto (3)'],
    cmap='viridis',
    normalize='true'
)
disp_rn.ax_.set_title("Matriz de Confusão - Teste Cego (Rede Neural)")
plt.show()