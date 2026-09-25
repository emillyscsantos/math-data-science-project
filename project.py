import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------
# 1. GERAÇÃO E PRÉ-PROCESSAMENTO DO DATASET
# ----------------------------------------------------------------------
def gera_dados(n_samples=600, noise=0.08, factor=0.5, seed=42):
    """
    Gera o dataset de círculos concêntricos e aplica a divisão treino/teste + normalização.
    """
    X, Y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=seed)
    Y = Y.reshape(-1, 1)  # Converte para matriz coluna (N, 1)

    # Divisão 70% Treino e 30% Teste
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.30, random_state=seed, stratify=Y)

    # Padronização (Ajusta o scaler apenas no treino para evitar data leakage)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return X_tr, X_te, Y_tr, Y_te

# ----------------------------------------------------------------------
# 2. FUNÇÕES DE ATIVAÇÃO E SUAS DERIVADAS
# ----------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# ----------------------------------------------------------------------
# 3. INICIALIZAÇÃO DOS PESOS (HE NORMAL)
# ----------------------------------------------------------------------
def inicializa_pesos(camadas, seed=42):
    """
    camadas: [n_in, n_hidden, n_out] -> Exemplo: [2, 8, 1]
    """
    np.random.seed(seed)
    W, b = [], []
    for i in range(len(camadas) - 1):
        n_in, n_out = camadas[i], camadas[i + 1]
        # Inicialização He Normal para ReLU na camada oculta / Xavier na saída
        std = np.sqrt(2.0 / n_in) if i < len(camadas) - 2 else np.sqrt(1.0 / n_in)
        w_camada = np.random.randn(n_in, n_out) * std
        b_camada = np.zeros((1, n_out))
        W.append(w_camada)
        b.append(b_camada)
    return W, b

# ----------------------------------------------------------------------
# 4. FORWARD PASS (PROPAGAÇÃO PARA A FRENTE)
# ----------------------------------------------------------------------
def forward(X, W, b):
    Zs, As = [], [X]
    n_camadas = len(W)

    for i in range(n_camadas):
        Z = As[-1] @ W[i] + b[i]
        A = sigmoid(Z) if i == n_camadas - 1 else relu(Z)
        Zs.append(Z)
        As.append(A)

    return Zs, As

# ----------------------------------------------------------------------
# 5. BACKWARD PASS (RETROPROPAGAÇÃO VETORIZADA)
# ----------------------------------------------------------------------
def backward(Y, W, b, Zs, As):
    n_camadas = len(W)
    N = Y.shape[0]  # Número de amostras

    e = As[-1] - Y
    loss = np.mean(0.5 * (e ** 2))

    grad_W = [None] * n_camadas
    grad_b = [None] * n_camadas

    grad_A = e / N

    for i in reversed(range(n_camadas)):
        if i == n_camadas - 1:
            grad_Z = grad_A * As[i + 1] * (1 - As[i + 1])
        else:
            grad_Z = grad_A * relu_derivative(Zs[i])

        grad_W[i] = As[i].T @ grad_Z
        grad_b[i] = grad_Z.sum(axis=0, keepdims=True)
        grad_A = grad_Z @ W[i].T

    return grad_W, grad_b, loss

# ----------------------------------------------------------------------
# 6. VERIFICAÇÃO NUMÉRICA DO GRADIENTE (DIFERENÇAS FINITAS)
# ----------------------------------------------------------------------
def verifica_gradiente(X, Y, W, b, h=1e-5):
    Zs, As = forward(X, W, b)
    grad_W, _, _ = backward(Y, W, b, Zs, As)

    c, i, j = 0, 0, 0
    w_original = W[c][i, j]

    W[c][i, j] = w_original + h
    _, As_mais = forward(X, W, b)
    loss_mais = np.mean(0.5 * (As_mais[-1] - Y) ** 2)

    W[c][i, j] = w_original - h
    _, As_menos = forward(X, W, b)
    loss_menos = np.mean(0.5 * (As_menos[-1] - Y) ** 2)

    W[c][i, j] = w_original

    grad_numerico = (loss_mais - loss_menos) / (2 * h)
    grad_analitico = grad_W[c][i, j]

    erro_relativo = abs(grad_numerico - grad_analitico) / max(1e-8, abs(grad_numerico) + abs(grad_analitico))
    return erro_relativo

# ----------------------------------------------------------------------
# 7. VALIDAÇÃO DOS GRADIENTES VIA AUTOGRAD DO PYTORCH
# ----------------------------------------------------------------------
def valida_gradientes_pytorch(X, Y, W, b):
    """
    Copia os pesos da implementação manual para um modelo PyTorch e
    compara param.grad com o backward manual em dupla precisão (float64).
    """
    x_sample = X[0:1]
    y_sample = Y[0:1]

    # Backward manual
    Zs, As = forward(x_sample, W, b)
    grad_W_manual, grad_b_manual, _ = backward(y_sample, W, b, Zs, As)

    # Modelo PyTorch correspondente (2 -> 8 -> 1)
    modelo_torch = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    ).double()

    with torch.no_grad():
        modelo_torch[0].weight.copy_(torch.tensor(W[0].T, dtype=torch.float64))
        modelo_torch[0].bias.copy_(torch.tensor(b[0].ravel(), dtype=torch.float64))
        modelo_torch[2].weight.copy_(torch.tensor(W[1].T, dtype=torch.float64))
        modelo_torch[2].bias.copy_(torch.tensor(b[1].ravel(), dtype=torch.float64))

    x_t = torch.tensor(x_sample, dtype=torch.float64)
    y_t = torch.tensor(y_sample, dtype=torch.float64)

    modelo_torch.zero_grad()
    y_pred = modelo_torch(x_t)
    loss_torch = 0.5 * torch.mean((y_pred - y_t) ** 2)
    loss_torch.backward()

    # Comparação usando np.allclose
    grad_w1_torch = modelo_torch[0].weight.grad.detach().numpy().T
    grad_b1_torch = modelo_torch[0].bias.grad.detach().numpy().reshape(1, -1)

    confere_w1 = np.allclose(grad_W_manual[0], grad_w1_torch, atol=1e-8)
    confere_b1 = np.allclose(grad_b_manual[0], grad_b1_torch, atol=1e-8)

    max_diff_w1 = np.max(np.abs(grad_W_manual[0] - grad_w1_torch))

    print("=== Conferência dos gradientes contra o PyTorch ===")
    print(f"W1: allclose = {confere_w1} | máx. diferença = {max_diff_w1:.2e}")
    print(f"b1: allclose = {confere_b1}")
    print("===================================================\n")

# ----------------------------------------------------------------------
# 8. TREINAMENTO DA REDE
# ----------------------------------------------------------------------
def acuracia(X, Y, W, b):
    _, As = forward(X, W, b)
    predicoes = (As[-1] >= 0.5).astype(float)
    return np.mean(predicoes == Y)

def treina_rede(X_tr, Y_tr, X_te, Y_te, camadas, taxa=0.5, epocas=10000):
    W, b = inicializa_pesos(camadas)

    # Checagens de gradiente
    erro_grad = verifica_gradiente(X_tr, Y_tr, W, b)
    print(f"Erro relativo (Diferenças Finitas): {erro_grad:.2e}")
    valida_gradientes_pytorch(X_tr, Y_tr, W, b)

    historico_loss = []

    for epoca in range(epocas):
        Zs, As = forward(X_tr, W, b)
        grad_W, grad_b, loss = backward(Y_tr, W, b, Zs, As)

        for i in range(len(W)):
            W[i] -= taxa * grad_W[i]
            b[i] -= taxa * grad_b[i]

        historico_loss.append(loss)

        if epoca % 2000 == 0 or epoca == epocas - 1:
            acc_tr = acuracia(X_tr, Y_tr, W, b) * 100
            acc_te = acuracia(X_te, Y_te, W, b) * 100
            print(f"Época {epoca:5d} | Perda: {loss:.6f} | Acc Treino: {acc_tr:.2f}% | Acc Teste: {acc_te:.2f}%")

    return W, b, historico_loss

# ----------------------------------------------------------------------
# 9. PLOTAGEM DA FRONTEIRA DE DECISÃO E CURVA DE PERDA
# ----------------------------------------------------------------------
def plota_resultados(X_tr, Y_tr, X_te, Y_te, W, b, historico_loss):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Curva de Perda (Log Scale)
    axes[0].plot(historico_loss, color='tab:blue')
    axes[0].set_yscale('log')
    axes[0].set_title("Histórico da Perda (Escala Logarítmica)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda (MSE)")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # Gráfico 2: Fronteira de Decisão
    x_min, x_max = X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5
    y_min, y_max = X_tr[:, 1].min() - 0.5, X_tr[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid = np.c_[xx.ravel(), yy.ravel()]
    _, As_grid = forward(grid, W, b)
    Z_grid = (As_grid[-1] >= 0.5).astype(float).reshape(xx.shape)

    axes[1].contourf(xx, yy, Z_grid, alpha=0.3, cmap=plt.cm.Spectral)
    axes[1].scatter(X_tr[:, 0], X_tr[:, 1], c=Y_tr.ravel(), cmap=plt.cm.Spectral, edgecolors='k', label='Treino')
    axes[1].scatter(X_te[:, 0], X_te[:, 1], c=Y_te.ravel(), cmap=plt.cm.binary, marker='x', alpha=0.6, label='Teste')
    axes[1].set_title("Fronteira de Decisão (Rede 2->8->1 com ReLU)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 10. EXECUÇÃO DO EXPERIMENTO E RELATÓRIO FINAL
# ----------------------------------------------------------------------
X_tr, X_te, Y_tr, Y_te = gera_dados(n_samples=800, noise=0.08, seed=42)
camadas_arquitetura = [2, 8, 1]

W, b, historico = treina_rede(X_tr, Y_tr, X_te, Y_te, camadas_arquitetura, taxa=0.5, epocas=10000)

# Avaliação Detalhada no Conjunto de Teste
_, As_te = forward(X_te, W, b)
y_pred_te = (As_te[-1] >= 0.5).astype(int)

print("\n=== MATRIZ DE CONFUSÃO (CONJUNTO DE TESTE) ===")
print(confusion_matrix(Y_te, y_pred_te))
print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(Y_te, y_pred_te, digits=3))

plota_resultados(X_tr, Y_tr, X_te, Y_te, W, b, historico)