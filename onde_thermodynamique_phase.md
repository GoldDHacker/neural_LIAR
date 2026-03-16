# Onde thermodynamique de phase (LIAR)

## But
L’onde de phase est un terme d’énergie additionnel injecté dans le champ du neurone afin de capter des **structures globales cycliques** (ex: parité), difficiles à représenter avec des interactions strictement locales (linéaire / bilinéaire / trilinéaire) sans explosion combinatoire.

L’idée est d’ajouter une « signature de phase » dépendant d’une **charge globale** (somme sur les entrées perçues/modulées) puis de la mélanger au champ énergétique via une porte apprise.

---

## Où elle vit dans le code
Fichier:
- `Logical Isring-Attractor with Relational-Attention/liar_neuron.py`

Paramètres concernés (par neurone / par sortie `out_features`):
- `phase_freq_base` (appris)
- `phase_freq_adapt` (appris)
- `phase_shift` (appris)
- `wave_gate_raw` (appris)
- `lambda_wave_raw` (appris, régularisation de la porte d’onde)

---

## Définition mathématique (niveau “champ énergétique”)

### 1) Entrée perçue et modulée
Le neurone transforme d’abord l’entrée via une perception (Box–Cox) puis applique une rétroaction fractale par l’état interne `Z`:

- Perception:

```text
x_log = perception(x)
```

- Modulation fractale à l’étape `t`:

```text
x_mod(t) = x_log + Linear(Z(t), W_retro)
```

### 2) Charge globale
On définit une charge scalaire par exemple comme la somme sur les composantes de `x_mod`:

```text
charge_sum(t) = Σ_i x_mod_i(t)
```

Dans le code:

```text
charge_sum = sum(x_mod, dim=features)
```

### 3) Onde de phase
L’onde est une cosinusoïde évaluée sur cette charge:

```text
E_wave(t) = cos( charge_sum(t) * phase_freq(t) + phase_shift )
```

Contraintes/paramétrisation:
- `phase_freq(t)` est **dynamique** et dépend de l’état attracteur `Z(t)`.

Dans le code (forme conceptuelle):

```text
phase_freq(t) = softplus( phase_freq_base + Linear(Z(t), phase_freq_adapt) )
```

Donc **phase_freq(t) > 0**.

### 4) Injection dans le champ total
Le champ total (linéaire + tenseur + tri éventuel) reçoit le terme d’onde via une porte:

```text
E_total(t) ← E_total(t) + wave_gate * E_wave(t)
```

avec:

```text
wave_gate = sigmoid(wave_gate_raw) ∈ (0, 1)
```

---

## Interaction avec la dynamique d’attracteur (Ising)
Le neurone itère une dynamique de type Ising / relaxation:

```text
Z(t+1) = tanh( beta * Z(t) + E_total(t) )
```

avec:

```text
beta = softplus(beta_raw) > 0
```

Rôle intuitif:
- `E_total` apporte l’évidence (dont l’onde)
- `beta` règle la “température” / force de bifurcation (plus beta est grand, plus `Z` se fige en ±1)

---

## Régularisation propre à l’onde (pénalité dédiée)
L’onde a sa pénalité dédiée, séparée de la pénalité structurelle générale.

Dans `LIARNeuron.get_wave_penalty()`:

### 1) Paramètre de pénalité appris
On utilise une paramétrisation positive simple:

```text
lambda_wave = softplus(lambda_wave_raw)
```

### 2) Pénalité sur l’activation moyenne de l’onde
On pénalise une activation moyenne trop élevée de la porte d’onde:

```text
wg_mean = mean(wave_gate)
wave_penalty = (wg_mean^2) * lambda_wave + eta * (lambda_wave_raw^2)
```

Constantes:
- `eta = 1e-3`

Interprétation:
- terme `(wg_mean^2) * lambda_wave` : pousse à garder l’onde off sauf besoin
- terme `eta * (lambda_wave_raw^2)` : régularise directement le paramètre brut (évite des valeurs extrêmes)

---

## Pénalité structurelle et coexistence avec l’onde
La pénalité structurelle générale (normes des poids linéaires/tensoriels/tri) vit dans `get_structural_penalty()`.

Une règle de coexistence est implémentée:
- la pénalité sur `W_lin` est modulée continûment par l’activité moyenne de la porte d’onde:

```text
wlin_penalty <- wlin_penalty * (1 - clamp(mean(wave_gate), 0, 1))
```

Note:
- Ceci concerne la pénalité structurelle; l’onde garde sa pénalité dédiée (`get_wave_penalty()`).

---

## Ce que l’onde apprend concrètement
L’onde permet au neurone de représenter des fonctions où la décision dépend d’une **classe modulo** d’une quantité globale.

Exemple conceptuel:
- Pour la parité, le label dépend d’un bit global “pair/impair”, qui est naturellement lié à un comportement périodique.
- `cos(ω * charge + φ)` fournit une base périodique compacte, et `wave_gate` décide si cette base est utile.

---

## Paramètres à suivre (diagnostics)
Sans instrumenter l’historique interne `Z`, les signaux les plus informatifs pour savoir si l’onde “démarre” sont:
- `wave_gate` (activation de l’onde)
- `phase_freq` (fréquence apprise)
- `lambda_wave` (force de régularisation apprise)
- `beta` (régime thermodynamique: chaud vs gel)

---

## Problèmes rencontrés en pratique

### 1) `struct_coeff` peut étouffer l’onde
Symptôme typique en apprentissage de la parité (et tâches similaires): dès que `struct_coeff` est non-nul, même très petit, il arrive que le modèle converge vers un régime “simple” où:

- `wave_gate` reste proche de 0 (l’onde reste off)
- `phase_freq` reste bas
- l’accuracy reste proche du hasard (≈ 0.50)

Interprétation: la pénalité structurelle favorise des solutions à faible complexité (petits poids), ce qui peut empêcher l’apparition du régime cyclique global nécessaire à la parité.

### 2) Interaction pénalité structurelle vs champ linéaire (`W_lin`)
Dans la pratique, une partie du problème venait du fait que la compression du champ linéaire pouvait empêcher l’onde de s’installer. Une règle de coexistence a été introduite dans `get_structural_penalty()`:

- la pénalité sur `||W_lin||` est modulée continûment par `1 - mean(wave_gate)`

Objectif: éviter que la compression linéaire “casse” le régime onde au moment où il commence à s’activer.

### 3) Pénalité de l’onde: risque d’extinction si mal paramétrée
Un mauvais choix de forme de pénalité peut forcer l’extinction de l’onde par simple pression d’optimisation (c.-à-d. apprendre à mettre `wave_gate ≈ 0` partout). Pour cette raison:

- la pénalité est appliquée à `mean(wave_gate)`
- elle est quadratique en `wg_mean` (faible près de 0)
- `lambda_wave` est appris avec une paramétrisation positive simple, et `lambda_wave_raw` est régularisé en L2

### 4) Sensibilité multi-seed (amorcage de l’onde)
Même sans pression structurelle (`struct_coeff = 0`), il existe une sensibilité à l’initialisation: certains seeds convergent naturellement vers un régime “onde active”, d’autres restent dans un régime sans onde.

Signaux observables typiques d’un seed qui “n’allume pas” l’onde:

- `wave_gate` reste très faible
- `phase_freq` ne monte pas
- `beta` peut rester bas (régime chaud qui ne bifurque pas vers une logique stable)
- accuracy ≈ 0.50

Interprétation: l’onde et la dynamique d’attracteur forment un système non-linéaire avec des bassins d’attraction; l’amorcage peut être raté selon l’initialisation.

### 5) Point important: ces problèmes sont distincts
- Le problème `struct_coeff` concerne la **pression de parcimonie** qui peut empêcher l’onde d’émerger.
- Le problème multi-seed (même à `struct_coeff = 0`) concerne l’**amorçage/bassin d’attraction** et ne se résout pas uniquement en ajustant la régularisation.

---

## Résumé en une équation (vue d’ensemble)
À chaque itération d’attracteur:

```text
phase_freq = softplus( phase_freq_base + Linear(Z, phase_freq_adapt) )
E_wave = cos( (Σ_i x_mod_i) * phase_freq + phase_shift )
E_total ← E_base + sigmoid(wave_gate_raw) * E_wave
Z ← tanh( softplus(beta_raw) * Z + E_total )
```

avec une régularisation séparée:

```text
wave_penalty = mean(sigmoid(wave_gate_raw))^2 * softplus(lambda_wave_raw) + eta * (lambda_wave_raw^2)
```
