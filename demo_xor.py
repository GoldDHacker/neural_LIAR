import os
import torch
import torch.nn as nn
import torch.optim as optim
from liar_neuron import LIARNeuron

def train_xor():
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    torch.manual_seed(seed)  # Pour la reproductibilite
    
    # Dataset XOR : Non-linéairement séparable (le tortionnaire du perceptron)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[-1.0], [1.0], [1.0], [-1.0]])
    
    # -------------------------------------------------------------
    # LA REVOLUTION DE L'ARCHITECTURE
    # Un SEUL neurone LIAR ! Pas de couche cachee, pas de spawns infinis.
    # On ajoute 4 dimensions latentes pour qu'il decouvre lui-meme son rang !
    model = LIARNeuron(in_features=2, out_features=1, max_steps=4, latent_dim=4)
    
    # L'Optimiseur Adam, plus fluide pour la retropropagation a travers le temps (BPTT)
    optimizer = optim.Adam(model.parameters(), lr=0.08)
    
    # La sortie est ecrasee entre [-1, 1] par l'attracteur tanh
    criterion = nn.MSELoss()
    
    print("==========================================================")
    print(" DEMONSTRATION : Neurone L.I.A.R sur le circuit XOR")
    print("==========================================================")
    print(f" Seed = {seed}")
    print("Objectif : Mettre en evidence qu'UN SEUL neurone tensoriel")
    print("peut plier l'espace logique sans couches cachees.")
    print("----------------------------------------------------------\n")
    
    epochs = 400
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Le neurone iteratif cherche son attracteur
        output = model(X)
        
        # La perte absolue : Erreur Logique + Friction (Parcimonie L2)
        mse_loss = criterion(output, Y)
        penalty = 0.001 * model.get_structural_penalty()
        loss = mse_loss + penalty
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            pred_class = (output > 0).float() * 2 - 1
            acc = (pred_class == Y).float().mean().item()
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss = {loss.item():.5f} | Acc = {acc*100:3.0f}%")

    print("\n==========================================================")
    print(" VERIFICATION RIGOUREUSE DES CRITERES (COMME VERSION A/B)")
    print("==========================================================")
    
    with torch.no_grad():
        final_out, history = model(X, return_history=True)
        acc = (((final_out > 0).float() * 2 - 1) == Y).float().mean().item()
        final_mse = criterion(final_out, Y).item()
        
        print("\n[TEST 1 : XOR_GROWTH (Capacite Structurelle)]")
        print("Objectif : Le modele peut-il resoudre XOR a 100% de precision ?")
        print(f"-> Precision atteinte : {acc*100:3.0f}%")
        if acc == 1.0:
            print("-> [SUCCES] L'architecture Tensorielle (Sigma-Pi) suffit a plier l'espace logique sans 'spawner' de neurones caches.")
        else:
            print("-> [ECHEC] Le reseau n'a pas pu croiser la topologie requise.")
            
        print("\n[TEST 2 : XOR_STRICT (Saturation Thermodynamique Absolue)]")
        print("Objectif : L'attracteur a-t-il vraiment 'bifurque' ? La prediction ne doit pas etre juste positive/negative, elle doit s'effondrer vers +1/-1 (|y_pred| > 0.8). La MSE doit etre quasi-nulle.")
        print(f"-> MSE Finale : {final_mse:.5f}")
        
        all_strict = True
        for i in range(4):
            x_val = [int(v) for v in X[i].tolist()]
            y_true = int(Y[i].item())
            y_pred = final_out[i].item()
            
            is_strict = abs(y_pred) >= 0.8
            if not is_strict:
                all_strict = False
            
            status = "OK (SATURÉ)" if is_strict else "FAIL (INCERTAIN)"
            print(f"  Entrée: {x_val} | Vrai: {y_true:2d} | Attracteur: {y_pred:+6.3f} -> {status}")
            
        if all_strict and final_mse < 0.05:
            print("-> [SUCCES] Le Moteur d'Ising a cree une brisure absolue. Le modele ne devine pas, il SAIT.")
        else:
            print("-> [ECHEC] L'attracteur est mou. La thermodynamique n'a pas atteint le point critique.")

        # ASSERTIONS FATALES (Si ces lignes ne levent pas d'erreur, le code est valide)
        assert acc == 1.0, "Echec de XOR_GROWTH : Precision < 100%"
        assert all_strict and final_mse < 0.05, "Echec de XOR_STRICT : Attracteur non sature ou MSE > 0.05"

    print("\n[TOPOLOGIE INTERNE DU NEURONE]")
    print("  Champ Linéaire (W_lin) :", ["{:.2f}".format(v) for v in model.W_lin.detach().numpy().flatten()])
    
    gate = model.tensor_gate.item()
    print(f"  Gating Tensoriel Actif : {gate*100:.1f}%")
    
    if model.latent_dim is not None:
        print(f"  Espace Tensoriel (Factorisation Latente D={model.latent_dim}) :")
        s = model.S_raw.detach().numpy()[0]
        s_formatted = [f"{v:+.3f}" for v in s]
        print(f"    Valeurs Singulières (S) : {s_formatted}")
        print("    -> Le L1 a éteint les dimensions inutiles pour trouver le rang mathématique exact.")
    else:
        t_mat = model.W_tensor.detach().numpy()[0]
        print("  Champ Tensoriel (W_tensor) :")
        print(f"  [[{t_mat[0,0]:+5.2f}, {t_mat[0,1]:+5.2f}]")
        print(f"   [{t_mat[1,0]:+5.2f}, {t_mat[1,1]:+5.2f}]]")
    print(f"  Température d'Ising (Beta) : {model.beta.detach().item():.3f}")
    alphas = [f"{a:.3f}" for a in model.alpha.detach().numpy()]
    print(f"  Puissance de Perception (Alpha) : {alphas} (-> 0 = Log, -> 1 = Linéaire)")

if __name__ == '__main__':
    train_xor()
