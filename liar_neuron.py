import torch
import torch.nn as nn
import torch.nn.functional as F

class LIARNeuron(nn.Module):
    """
    Logical Ising-Attractor with Relational-Attention (L.I.A.R)
    
    Le Système 1 : Reconnaissance Logique Instantanée de Motifs.
    
    Unifie 4 concepts fondamentaux :
    1. Perception Logarithmique : Transforme les proportions multiplicatives en differences additives.
    2. Interactions Tensorielles (Sigma-Pi Factorise) : Capture les relations logiques d'ordre superieur sans explosion combinatoire.
    3. Energie de Champ : Accumulation de l'evidence lineaire et relationnelle.
    4. Dynamique d'Ising : Restauration de la brisure de symetrie par saturation d'auto-retroaction positive (tanh).
    
    NOTE : Ce neurone est un "Déducteur Balistique" spatial. Il ne gere ni la
    memoire temporelle, ni le raisonnement sequentiel, ni l'arret autonome.
    Pour ces capacites, voir H.O.R.A. (Homotopic Operator with Recurrent Attractors).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        max_steps: int = 5,
        latent_dim: int = None,
        max_order: int = 3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_steps = max_steps
        self.latent_dim = latent_dim
        self.max_order = max_order
        
        # 1. Le Champ Lineaire (L'evidence simple : apprentissage type Perceptron classique)
        self.W_lin = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.b = nn.Parameter(torch.zeros(out_features))
        
        # 2. L'Interaction Tensorielle (La logique de dependance : XOR, AND, etc.)
        if latent_dim is None:
            # Mode "Brute Force" Tensoriel (Pour petites dimensions)
            self.W_tensor = nn.Parameter(torch.randn(out_features, in_features, in_features) * 0.1)
        else:
            # Mode "Facteur Latent" (Low-Rank factorized pour eviter l'explosion combinatoire)
            self.U = nn.Parameter(torch.randn(out_features, in_features, latent_dim) * 0.1)  # Emetteur
            self.V = nn.Parameter(torch.randn(out_features, in_features, latent_dim) * 0.1)  # Recepteur
            self.S_raw = nn.Parameter(torch.ones(out_features, latent_dim) * 0.5)            # Valeurs Singulieres
            
        # 2.5 La Porte Tensorielle de Contexte (Le neurone décide "Est-ce que j'ai besoin du tenseur ?")
        self.tensor_gate_raw = nn.Parameter(torch.zeros(out_features))

        if self.max_order >= 3:
            if latent_dim is None:
                tri_dim = in_features
            else:
                tri_dim = latent_dim
            self.A3 = nn.Parameter(torch.randn(out_features, in_features, tri_dim) * 0.1)
            self.B3 = nn.Parameter(torch.randn(out_features, in_features, tri_dim) * 0.1)
            self.C3 = nn.Parameter(torch.randn(out_features, in_features, tri_dim) * 0.1)
            self.S3_raw = nn.Parameter(torch.ones(out_features, tri_dim) * 0.5)

            self.g1_raw = nn.Parameter(torch.ones(out_features) * 2.0)
            self.g2_raw = nn.Parameter(torch.zeros(out_features))
            self.g3_raw = nn.Parameter(torch.ones(out_features) * -2.0)
            
        # 2.9 Onde Thermodynamique de Phase (Pour la Parite globale)
        # Aucune bequille, aucun bruit aleatoire. Initialisation a l'equilibre absolu (0).
        self.phase_freq_base = nn.Parameter(torch.zeros(out_features))
        self.phase_freq_adapt = nn.Parameter(torch.zeros(out_features, out_features)) # Permet a l'onde de s'adapter dynamiquement a Z
        self.phase_shift = nn.Parameter(torch.zeros(out_features))
        self.wave_gate_raw = nn.Parameter(torch.ones(out_features) * -6.0) # Eteint a l'equilibre initial
        self.lambda_wave_raw = nn.Parameter(torch.tensor(0.0))
        
        # 2.8 Retroaction Fractale : le neurone module sa perception par sa croyance
        # W_retro permet a Z (la croyance courante) de modifier X_log avant le tenseur.
        # Initialise a zero pour que l'etape 0 soit identique au comportement classique.
        # Quand le reseau apprend, W_retro != 0 permet a chaque iteration d'Ising
        # de croiser des ordres de plus en plus eleves (fractale).
        self.W_retro = nn.Parameter(torch.zeros(in_features, out_features))
        
        # 3. La Température d'Ising (Force de la bifurcation / Auto-retroaction)
        # beta regule la force avec laquelle l'attracteur s'effondre dans un etat +1 ou -1.
        self.beta_raw = nn.Parameter(torch.zeros(out_features))
        
        # 4. L'exposant d'Echelle de la Perception (Alpha)
        # Permet au neurone d'apprendre son propre degre de compression (au lieu d'un log fixe)
        # Base de Box-Cox : si alpha -> 0 on retrouve le log pur.
        # Initialisation proche de 0 (0.1) pour commencer avec un comportement hyper-compressif.
        self.alpha_raw = nn.Parameter(torch.ones(in_features) * -2.0)
        
    @property
    def tensor_gate(self):
        # 0.0 -> 1.0 (Soft gate global sur l'utilisation du tenseur)
        return torch.sigmoid(self.tensor_gate_raw)

    @property
    def g1(self):
        if not hasattr(self, "g1_raw"):
            return torch.ones_like(self.tensor_gate_raw)
        return torch.sigmoid(self.g1_raw)

    @property
    def g2(self):
        if not hasattr(self, "g2_raw"):
            return torch.ones_like(self.tensor_gate_raw)
        return torch.sigmoid(self.g2_raw)

    @property
    def g3(self):
        if not hasattr(self, "g3_raw"):
            return torch.zeros_like(self.tensor_gate_raw)
        return torch.sigmoid(self.g3_raw)

    @property
    def wave_gate(self):
        return torch.sigmoid(self.wave_gate_raw)
        
    def dynamic_phase_freq(self, Z: torch.Tensor):
        # La frequence s'adapte dynamiquement en fonction de l'etat d'attraction (Z)
        # Resonance Parametrique : La "tension" du neurone modifie sa frequence de vibration
        freq_raw = self.phase_freq_base + F.linear(Z, self.phase_freq_adapt)
        return F.softplus(freq_raw) # (B, out_features)

    @property
    def beta(self):
        # Beta commence à ~0.69 (chaud) et peut grandir pour geler la logique
        return F.softplus(self.beta_raw)

    @property
    def alpha(self):
        # Alpha doit etre strictement positif. 
        # F.softplus(-2.0) donne ~0.12, un depart proche du logarithme
        return F.softplus(self.alpha_raw) + 1e-4

    def perception(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforme l'espace physique multiplicatif en espace additif via Box-Cox.
        Si alpha -> 0 : x_log = log(|x| + 1) * sign(x) (Limite de L'Hopital)
        Si alpha == 1 : x_log = x (Lineaire)
        """
        alpha = self.alpha.unsqueeze(0) # Broadcasting pour le batch (1, in_features)
        magnitude = torch.abs(x) + 1.0
        x_percu = torch.sign(x) * (torch.pow(magnitude, alpha) - 1.0) / alpha
        return x_percu

    def forward(self, x: torch.Tensor, return_history: bool = False):
        """
        Calcule l'etat final de l'attracteur thermodynamique par integration temporelle discrete.
        x: (Batch, in_features)
        """
        B = x.size(0)
        
        # --- ETAPE 1 : PERCEPTION LOGARITHMIQUE ---
        x_log = self.perception(x)
        
        # --- ETAPE 2 : CALCUL DE L'ENERGIE DU CHAMP FIXE (W_lin) ---
        E_lin = F.linear(x_log, self.W_lin, self.b)  # (Batch, out_features)
        
        # --- ETAPE 3 : DYNAMIQUE D'ATTRACTEUR D'ISING (FRACTALE) ---
        Z = torch.zeros(B, self.out_features, device=x.device)
            
        history = []
        
        for step in range(self.max_steps):
            
            # 3.1 RETROACTION FRACTALE ADDITIVE : l'attention dirige le focus
            # x_mod = x_log + Z @ W_retro^T
            # L'addition preserve parfaitement le gradient à travers le temps (ResNet-like).
            # Z agit comme un "biais d'attention" dynamique sur l'espace d'entree.
            x_mod = x_log + F.linear(Z, self.W_retro)  # (B, in_features)
            
            # 3.2 Interaction Tensorielle sur la perception modulee
            if self.latent_dim is None:
                E_tensor = torch.einsum('bi,oij,bj->bo', x_mod, self.W_tensor, x_mod)
            else:
                Q = torch.einsum('bi,oid->bod', x_mod, self.U)
                K = torch.einsum('bi,oid->bod', x_mod, self.V)
                E_tensor = torch.sum(Q * K * self.S_raw.unsqueeze(0), dim=-1)
                
            if self.max_order >= 3:
                a3 = torch.einsum('bi,oid->bod', x_mod, self.A3)
                b3 = torch.einsum('bi,oid->bod', x_mod, self.B3)
                c3 = torch.einsum('bi,oid->bod', x_mod, self.C3)
                E_tri = torch.sum(a3 * b3 * c3 * self.S3_raw.unsqueeze(0), dim=-1)

                E_total = self.g1.unsqueeze(0) * E_lin + self.g2.unsqueeze(0) * E_tensor + self.g3.unsqueeze(0) * E_tri
            else:
                E_total = E_lin + self.tensor_gate.unsqueeze(0) * E_tensor

            # 3.2.5 Onde de Superposition (Evaluation de la Phase / Parite Globale)
            # detecte si la "charge" d'entree est cycliquement synchrone (ex: Parite absolue)
            charge_sum = torch.sum(x_mod, dim=1, keepdim=True) # (B, 1)
            
            # La frequence s'adapte DYNAMIQUEMENT a chaque pas de temps selon la croyance (Z)
            freq_t = self.dynamic_phase_freq(Z) # (B, out_features)
            E_wave = torch.cos(charge_sum * freq_t + self.phase_shift.unsqueeze(0))
            
            E_total = E_total + self.wave_gate.unsqueeze(0) * E_wave

            # 3.3 Formule Universelle : z_{t+1} = tanh(beta * z_t + champ_energetique)
            Z = torch.tanh(self.beta * Z + E_total)
            if return_history:
                history.append(Z.clone())
                
        if return_history:
            return Z, history
        return Z
        
    def get_structural_penalty(self):
        """
        Penalite thermodynamique pour forcer le modele a developper des 
        structures logiques parcimonieuses plutot que de crier sur le bruit.
        """
        penalty = torch.tensor(0.0, device=self.W_lin.device)
        if self.latent_dim is None:
            penalty = penalty + torch.norm(self.W_tensor)
        else:
            penalty = penalty + torch.norm(self.U) + torch.norm(self.V) + 2.0 * torch.norm(self.S_raw, p=1)

        if self.max_order >= 3:
            penalty = penalty + torch.norm(self.A3) + torch.norm(self.B3) + torch.norm(self.C3) + 2.0 * torch.norm(self.S3_raw, p=1)
        return penalty

    def get_wave_penalty(self):
        penalty = torch.tensor(0.0, device=self.W_lin.device)
        if hasattr(self, "wave_gate_raw"):
            eta = 1e-3
            lambda_wave = F.softplus(self.lambda_wave_raw)
            wg_mean = torch.mean(self.wave_gate)
            penalty = penalty + (wg_mean ** 2) * lambda_wave + eta * (self.lambda_wave_raw ** 2)
        return penalty
