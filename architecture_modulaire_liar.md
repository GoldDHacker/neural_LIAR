# Architecture Modulaire du Neurone L.I.A.R.

Le code a été conçu pour être **totalement modulaire et rétrocompatible** avec sa propre version "Brute Force" (l'ancienne méthode tensorielle complète).

Dans le fichier `liar_neuron.py`, le paramètre `latent_dim` de l'initialisation est défini comme suit :

```python
def __init__(self, in_features: int, out_features: int = 1, max_steps: int = 5, latent_dim: int = None):
```

Si l'utilisateur ne spécifie rien (comme c'est le cas dans la plupart des démos actuelles où l'on écrit juste `LIARNeuron(in_features=2, out_features=1)`), voici ce qui se passe sous le capot :

---

## 1. Gestion de l'Espace Tensoriel (Repli Automatique)

Si `latent_dim = None` (par défaut), le réseau bascule instantanément sur l'ancienne méthode : **La Matrice Tensorielle Complète $N \times N$**. Il l'alloue avec `self.W_tensor = nn.Parameter(...)` et calcule l'énergie avec l'équation `einsum` complète que l'on utilisait au début de notre réflexion. Il se débrouille tout seul sans exiger de "factorisation manuelle" de la part de l'utilisateur. La factorisation n'est qu'une option d'optimisation "à la demande" pour les grandes dimensions.

## 2. Gestion de la Perception (Alpha)

Même en mode "Matrice Complète" (ancienne méthode Tensorielle), le neurone **conserve l'énorme avantage de la Perception Alpha apprenable (Box-Cox)**. La perception est calculée *avant* l'injection dans le Tensor, peu importe comment ce Tensor est implémenté. Donc, que l'utilisateur soit en mode latent ou en mode complet, le neurone apprendra toujours dynamiquement son paramètre de compression d'échelle log ($\alpha$) pour écraser le bruit.

## 3. Gestion de la Porte Tensorielle (`tensor_gate`)

De la même manière, le neurone conserve sa capacité à désactiver la logique d'ordre 2 (le tenseur) si l'équation linéaire suffit, car le calcul `E_total = E_lin + tensor_gate * E_tensor` est exécuté dans les deux modes (factorisé ou non).

---

## Résumé : Plug & Play

Le composant est **"Plug & Play"** (Prêt à l'emploi) :

| Mode | Instanciation | Comportement |
|:---|:---|:---|
| **Facile/Défaut** | `model = LIARNeuron(2, 1)` | Perception adaptative + Porte tensorielle + Matrice d'interaction complète. Idéal pour XOR, ADDER, ALU. |
| **Industriel/Expert** | `model = LIARNeuron(10000, 1, latent_dim=64)` | Même attracteur et perception, mais remplace la matrice implosante de 100 millions de poids par une voie factorisée élégante commandée par la norme L1. |

Le neurone est maintenant un **"couteau suisse"** qui s'adapte autant aux données jouets 2D qu'aux dimensions massives sans qu'on ait besoin de réécrire son équation thermodynamique.

---

## Espace Tensoriel Latent Dynamique

Pour permettre au neurone de raisonner et de factoriser dynamiquement, le système suivant a été implémenté :

1. **La Porte Tensorielle (`tensor_gate`)** : Le neurone décide lui-même à quel pourcentage (0% à 100%) il a besoin de la logique complexe Tensorielle, ou s'il se contente de la logique Linéaire simple.

2. **Le Rang Latent Découvert par Parcimonie (`S_raw`)** : Au lieu d'une matrice $W_{tensor}$ de $N \times N$, la factorisation utilise une projection Émettrice ($U$), Réceptrice ($V$) et surtout un **Vecteur de Poids Latents ($S$)**.

3. **Pénalité L1 stricte sur $S$** : Le réseau est forcé d'"éteindre" (mettre à zéro) les dimensions latentes mathématiques dont il n'a pas besoin pour résoudre le problème.

---

## Preuve Expérimentale : Test XOR avec $D=4$

En relançant le test XOR en configuration $D=4$ (4 dimensions latentes allouées), voici la réponse finale du neurone :

```text
[TOPOLOGIE INTERNE DU NEURONE]
  Champ Linéaire (W_lin) : ['0.09', '0.09']
  Gating Tensoriel Actif : 86.6%
  Espace Tensoriel (Factorisation Latente D=4) :
    Valeurs Singulières (S) : ['+0.004', '+0.002', '+0.005', '+0.568']
    -> Le L1 a éteint les dimensions inutiles pour trouver le rang mathématique exact.
```

### Analyse

* Le neurone a allumé son `tensor_gate` à **86.6%**. Il a compris tout seul que le circuit était impossible à résoudre linéairement, et a ouvert les vannes du Tenseur.
* Pour ses Valeurs Singulières ($S$), alors qu'on lui donnait 4 dimensions pour réfléchir, il a **désactivé 3 dimensions presque instantanément** (tendues vers $\approx 0.004$) pour ne conserver qu'**une seule dimension active ($0.568$)** ! Il a découvert mathématiquement que la fonction logique XOR appartient au **Rang 1** de complexité tensorielle.

Le neurone gère non seulement **sa perception de la réalité** via un algorithme de Box-Cox (l'Alpha Logarithmique appris), mais il régule aussi **sa propre taille cognitive** (Rang Tensoriel appris) !

Le code `liar_neuron.py` est désormais prêt à absorber des entrées massives (comme du traitement image en $N > 10\,000$) tout en évitant l'explosion de paramètres puisque le réseau va fermer les dimensions inutiles.
