"""Exemple simple d=1 (regression linéaire) trouver le poids w"""



# Implémentation simple de FedSGD avec Local Differential Privacy (LDP)

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paramètres de l'expérience
# -----------------------------
K = 10                # Nombre d'utilisateurs
T = 100               # Nombre d'itérations
d = 1                 # Dimension du poids (simple regression)
eta = 0.1             # Taux d'apprentissage
C = 1.0               # Seuil de clippage
sigma = 1.0           # écart-type du bruit Gaussien
q = 0.5               # Probabilité d'échantillonnage

# -----------------------------
# Génération des données locales
# -----------------------------
def generate_local_data(n_samples=20):
    x = np.random.uniform(-5, 5, n_samples)
    y = 2 * x + np.random.normal(0, 1, n_samples)  # vraie relation y = 2x + bruit
    return x, y

local_data = [generate_local_data() for _ in range(K)]

# -----------------------------
# Initialisation du modèle global
# -----------------------------
w = np.random.randn(d)  # Poids initial aléatoire

# -----------------------------
# Fonction pour calculer le gradient local
# -----------------------------
def compute_local_gradient(x, y, w):
    preds = w[0] * x
    errors = preds - y
    grad = np.mean(2 * errors * x)
    return np.array([grad])

# -----------------------------
# Boucle d'entraînement Federated SGD avec LDP
# -----------------------------
losses = []

for t in range(1, T + 1):
    gradients = []

    for k in range(K):
        # Échantillonnage
        x_local, y_local = local_data[k]
        indices = np.random.rand(len(x_local)) < q
        x_sampled = x_local[indices]
        y_sampled = y_local[indices]

        if len(x_sampled) == 0:
            continue  # si aucun échantillon choisi

        # Calcul du gradient local
        grad = compute_local_gradient(x_sampled, y_sampled, w)

        # Clippage
        grad_norm = np.linalg.norm(grad)
        grad_clipped = grad / max(1, grad_norm / C)

        # Ajout de bruit Gaussien
        noise = np.random.normal(0, C * sigma, size=grad.shape)
        grad_noisy = grad_clipped + noise

        gradients.append(grad_noisy)

    if len(gradients) == 0:
        continue

    # Agrégation des gradients bruités
    mean_gradient = np.mean(gradients, axis=0)

    # Mise à jour des poids
    w = w - eta * mean_gradient

    # Calcul et stockage de la perte globale simulée
    total_loss = 0
    total_samples = 0
    for k in range(K):
        x_local, y_local = local_data[k]
        preds = w[0] * x_local
        errors = preds - y_local
        total_loss += np.sum(errors ** 2)
        total_samples += len(x_local)
    losses.append(total_loss / total_samples)

# -----------------------------
# Affichage du résultat
# -----------------------------
print(f"Poids final appris: {w}")

plt.plot(losses)
plt.xlabel("Itération")
plt.ylabel("Loss moyen global")
plt.title("Évolution de la perte globale pendant l'entraînement")
plt.grid()
plt.show()





"""d=2"""

# Implémentation FedSGD avec Local Differential Privacy (LDP) - Dimension 2

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paramètres de l'expérience
# -----------------------------
K = 10                # Nombre d'utilisateurs
T = 200               # Nombre d'itérations
d = 2                 # Dimension des poids (régression multivariée)
eta = 0.1             # Taux d'apprentissage
C = 1.0               # Seuil de clippage
sigma = 1.0           # écart-type du bruit Gaussien
q = 0.5               # Probabilité d'échantillonnage

# -----------------------------
# Génération des données locales
# -----------------------------
def generate_local_data(n_samples=20):
    X = np.random.uniform(-5, 5, (n_samples, d))  # Deux colonnes : x1 et x2
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 1, n_samples)  # y = 2x1 + 3x2 + bruit
    return X, y

local_data = [generate_local_data() for _ in range(K)]

# -----------------------------
# Initialisation du modèle global
# -----------------------------
w = np.random.randn(d)  # Poids initiaux aléatoires (2 valeurs)

# -----------------------------
# Fonction pour calculer le gradient local
# -----------------------------
def compute_local_gradient(X, y, w):
    preds = X @ w  # Produit matriciel (n échantillons)
    errors = preds - y
    grad = np.mean(2 * errors[:, None] * X, axis=0)  # Gradient pour chaque dimension
    return grad

# -----------------------------
# Boucle d'entraînement Federated SGD avec LDP
# -----------------------------
losses = []

for t in range(1, T + 1):
    gradients = []

    for k in range(K):
        # Échantillonnage
        X_local, y_local = local_data[k]
        indices = np.random.rand(len(X_local)) < q
        X_sampled = X_local[indices]
        y_sampled = y_local[indices]

        if len(X_sampled) == 0:
            continue  # Aucun échantillon sélectionné

        # Calcul du gradient local
        grad = compute_local_gradient(X_sampled, y_sampled, w)

        # Clippage
        grad_norm = np.linalg.norm(grad)
        grad_clipped = grad / max(1, grad_norm / C)

        # Ajout de bruit Gaussien
        noise = np.random.normal(0, C * sigma, size=grad.shape)
        grad_noisy = grad_clipped + noise

        gradients.append(grad_noisy)

    if len(gradients) == 0:
        continue

    # Agrégation des gradients bruités
    mean_gradient = np.mean(gradients, axis=0)

    # Mise à jour des poids
    w = w - eta * mean_gradient

    # Calcul et stockage de la perte globale simulée
    total_loss = 0
    total_samples = 0
    for k in range(K):
        X_local, y_local = local_data[k]
        preds = X_local @ w
        errors = preds - y_local
        total_loss += np.sum(errors ** 2)
        total_samples += len(X_local)
    losses.append(total_loss / total_samples)

# -----------------------------
# Affichage du résultat
# -----------------------------
print(f"Poids finaux appris: {w}")

plt.plot(losses)
plt.xlabel("Itération")
plt.ylabel("Loss moyen global")
plt.title("Évolution de la perte globale (d=2)")
plt.grid()
plt.show()




