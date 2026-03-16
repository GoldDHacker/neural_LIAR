import os
import torch
import torch.nn as nn
import torch.optim as optim
from liar_neuron import LIARNeuron

def test_noise_robustness():
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    torch.manual_seed(seed)
    
    # Dataset XOR
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[-1.0], [1.0], [1.0], [-1.0]])
    
    model = LIARNeuron(in_features=2, out_features=1, max_steps=4)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    
    print("==========================================================")
    print(" TEST DE ROBUSTESSE AU BRUIT (NOISE STRESS-TEST)")
    print("==========================================================")
    print(f" Seed = {seed}")
    print("Objectif : Entrainer sur du signal clair, et verifier si la brisure")
    print("de symetrie (Attracteur d'Ising) survit a un bruit massif non vu.")
    print("----------------------------------------------------------\n")
    
    epochs = 400
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y) + 0.001 * model.get_structural_penalty()
        loss.backward()
        optimizer.step()

    print("[EVALUATION SANS BRUIT (BASELINE)]")
    with torch.no_grad():
        Z_clean = model(X)
        acc_clean = (((Z_clean > 0).float() * 2 - 1) == Y).float().mean().item()
        print(f"  Accuracy Clean : {acc_clean*100:3.0f}%")
        
    print("\n[EVALUATION AVEC BRUIT (SIGMA = 0.3)]")
    # Ajout d'un bruit de 0.3 d'ecart type (30% du max du signal)
    noise_sigma = 0.3
    noise_trials = int(os.environ.get("THERMO_NOISE_TRIALS", "20"))
    noise_seed_env = os.environ.get("THERMO_NOISE_SEED", "")
    noise_seed = int(noise_seed_env) if noise_seed_env != "" else None
    if noise_seed is None:
        noise_seed = seed
    
    with torch.no_grad():
        acc_trials = []
        worst_acc = 1.0
        worst_X_noisy = None
        worst_Z_noisy = None

        for t in range(noise_trials):
            g = torch.Generator(device=X.device)
            g.manual_seed(noise_seed + t)
            X_noisy = X + torch.randn(X.shape, generator=g, device=X.device) * noise_sigma

            Z_noisy = model(X_noisy)
            acc_noisy = (((Z_noisy > 0).float() * 2 - 1) == Y).float().mean().item()
            acc_trials.append(acc_noisy)

            if acc_noisy < worst_acc:
                worst_acc = acc_noisy
                worst_X_noisy = X_noisy.clone()
                worst_Z_noisy = Z_noisy.clone()

        acc_mean = float(sum(acc_trials) / max(len(acc_trials), 1))
        acc_min = float(min(acc_trials))
        acc_max = float(max(acc_trials))

        print(f"  THERMO_NOISE_TRIALS = {noise_trials}")
        print(f"  THERMO_NOISE_SEED   = {noise_seed}")
        print(f"\n-> Accuracy Noisy (mean/min/max) : {acc_mean*100:3.0f}% / {acc_min*100:3.0f}% / {acc_max*100:3.0f}%")

        print("\n[DETAIL DU PIRE CAS (trial le plus dur)]")
        for i in range(4):
            x_clean_val = [int(v) for v in X[i].tolist()]
            x_noisy_val = [round(v, 2) for v in worst_X_noisy[i].tolist()]
            y_pred = worst_Z_noisy[i].item()
            is_strict = abs(y_pred) >= 0.7  # On tolere 0.7 avec du bruit lourd
            status = "RESISTE" if is_strict else "A FAIBLI"
            print(f"  Entrée {x_clean_val} baisée en {x_noisy_val} -> Attracteur: {y_pred:+6.3f} | {status}")

        if acc_mean >= 0.75:
            print("\n-> [SUCCES] Le modele s'est montre robuste au bruit (accuracy moyenne >= 75%).")
        else:
            print("\n-> [ECHEC] Modele trop fragile au bruit (accuracy moyenne < 75%).")

        assert acc_mean >= 0.75, "Echec du test de robustesse"

if __name__ == '__main__':
    test_noise_robustness()
