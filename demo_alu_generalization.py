"""
L.I.A.R. : Test de GENERALISATION Logique (ALU)
================================================
TEST CRUCIAL : Le neurone apprend-il a deduire une porte logique inconnue ?

PROTOCOLE :
  - ENTRAINEMENT : AND, OR, XOR (3 portes logiques, 12 exemples)
  - TEST OOD     : NAND (porte logique JAMAIS VUE, 4 exemples)

Si le neurone generalise, il a compris que NAND = NOT(AND), 
ce qui implique une comprehension structurelle de la negation logique.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from liar_neuron import LIARNeuron

def demo_alu_generalization():
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    torch.manual_seed(seed)
    
    # --- DATASET ---
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    labels_test = []
    
    for o1 in [0, 1]:
        for o2 in [0, 1]:
            for a in [0, 1]:
                for b in [0, 1]:
                    if o1 == 0 and o2 == 0:   res = a & b       # AND
                    elif o1 == 0 and o2 == 1: res = a | b       # OR
                    elif o1 == 1 and o2 == 0: res = a ^ b       # XOR
                    else:                     res = 1 - (a & b) # NAND
                    
                    x = [float(o1), float(o2), float(a), float(b)]
                    y = [float(res * 2 - 1)]
                    
                    if not (o1 == 1 and o2 == 1):  # Tout sauf NAND
                        X_train.append(x)
                        Y_train.append(y)
                    else:
                        X_test.append(x)
                        Y_test.append(y)
                        labels_test.append((a, b, res))
    
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    Y_te = torch.tensor(Y_test, dtype=torch.float32)
    
    model = LIARNeuron(in_features=4, out_features=1, max_steps=5)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    
    print("==========================================================")
    print(" L.I.A.R. : TEST DE GENERALISATION LOGIQUE (ALU)")
    print("==========================================================")
    print(f" Seed = {seed}")
    print(f" TRAIN : {len(X_train)} exemples (AND + OR + XOR)")
    print(f" TEST  : {len(X_test)} exemples (NAND) [JAMAIS VU]")
    print("----------------------------------------------------------\n")
    
    epochs = 1200
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tr)
        loss = criterion(output, Y_tr) + 0.001 * model.get_structural_penalty()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 300 == 0:
            with torch.no_grad():
                Z = model(X_tr)
                pred = (Z > 0).float() * 2 - 1
                acc = (pred == Y_tr).float().mean().item()
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss = {loss.item():.5f} | Train Acc = {acc*100:3.0f}%")
    
    # --- VERIFICATION TRAIN ---
    print("\n--- VERIFICATION SUR LE TRAIN (AND + OR + XOR) ---")
    with torch.no_grad():
        Z_train = model(X_tr)
        pred_train = (Z_train > 0).float() * 2 - 1
        acc_train = (pred_train == Y_tr).float().mean().item()
        print(f"[TRAIN] Precision : {acc_train*100:.1f}%")
    
    # --- TEST OOD : NAND ---
    print("\n--- TEST DE GENERALISATION : NAND [JAMAIS VU] ---")
    with torch.no_grad():
        Z_test = model(X_te)
        pred_test = (Z_test > 0).float() * 2 - 1
        acc_test = (pred_test == Y_te).float().mean().item()
        
        for i, (a, b, expected_bit) in enumerate(labels_test):
            y_pred = Z_test[i].item()
            expected_spin = expected_bit * 2 - 1
            is_correct = (y_pred > 0 and expected_spin > 0) or (y_pred < 0 and expected_spin < 0)
            status = "OK" if is_correct else "ERR"
            print(f"  NAND({a},{b}) = {expected_bit} (spin {expected_spin:+d}) | Pred: {y_pred:+.3f} | {status}")
        
        print(f"\n[GENERALISATION] Precision OOD (NAND) : {acc_test*100:.1f}%")
        
        if acc_test == 1.0:
            print("-> [SUCCES] Le neurone a deduit NAND = NOT(AND) sans l'avoir jamais vu !")
        elif acc_test >= 0.5:
            print("-> [PARTIEL] Deduction partielle de la negation logique.")
        else:
            print("-> [ECHEC] Le neurone ne generalise pas aux portes inconnues.")

if __name__ == '__main__':
    demo_alu_generalization()
