"""
DIAGNOSTIC 2 : Tester si l'encodage spin {-1,+1} des opcodes corrige NAND OOD
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from liar_neuron import LIARNeuron

def to_spin(bit):
    return float(bit) * 2.0 - 1.0

def demo_spin_opcodes():
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    torch.manual_seed(seed)
    
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    labels_test = []
    
    for o1 in [0, 1]:
        for o2 in [0, 1]:
            for a in [0, 1]:
                for b in [0, 1]:
                    if o1 == 0 and o2 == 0:   res = a & b
                    elif o1 == 0 and o2 == 1: res = a | b
                    elif o1 == 1 and o2 == 0: res = a ^ b
                    else:                     res = 1 - (a & b)
                    
                    # CHANGEMENT ICI : Opcodes en spins {-1, +1}
                    x = [to_spin(o1), to_spin(o2), to_spin(a), to_spin(b)]
                    y = [to_spin(res)]
                    
                    if not (o1 == 1 and o2 == 1):
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
    
    print("=" * 70)
    print(" TEST : Opcodes en SPIN {-1,+1} au lieu de {0,1}")
    print("=" * 70)
    
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
            print(f"  Epoch {epoch+1:4d}/{epochs} | Loss = {loss.item():.5f} | Train Acc = {acc*100:3.0f}%")
    
    # --- TEST OOD : NAND ---
    print("\n--- TEST DE GENERALISATION : NAND [JAMAIS VU] ---")
    with torch.no_grad():
        Z_test = model(X_te)
        for i, (a, b, expected) in enumerate(labels_test):
            y_pred = Z_test[i].item()
            expected_spin = expected * 2 - 1
            ok = "OK" if ((y_pred > 0) == (expected_spin > 0)) else "ERR"
            print(f"  NAND({a},{b}) = {expected} (spin {expected_spin:+d}) | Pred: {y_pred:+.3f} | {ok}")
        
        pred_test = (Z_test > 0).float() * 2 - 1
        acc_test = (pred_test == Y_te).float().mean().item()
        print(f"\n  [GENERALISATION OOD] Precision NAND : {acc_test*100:.1f}%")
    
    # --- TEST 2 : Plus d'epochs ---
    print("\n" + "=" * 70)
    print(" TEST 2 : Meme chose avec 2400 epochs + lr plus faible")
    print("=" * 70)
    
    torch.manual_seed(seed)
    model2 = LIARNeuron(in_features=4, out_features=1, max_steps=5)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.02)
    
    for epoch in range(2400):
        optimizer2.zero_grad()
        output = model2(X_tr)
        loss = criterion(output, Y_tr) + 0.001 * model2.get_structural_penalty()
        loss.backward()
        optimizer2.step()
    
    with torch.no_grad():
        Z_tr2 = model2(X_tr)
        pred_tr2 = (Z_tr2 > 0).float() * 2 - 1
        acc_tr2 = (pred_tr2 == Y_tr).float().mean().item()
        print(f"  Train Accuracy : {acc_tr2*100:.1f}%")
        
        Z_te2 = model2(X_te)
        for i, (a, b, expected) in enumerate(labels_test):
            y_pred = Z_te2[i].item()
            expected_spin = expected * 2 - 1
            ok = "OK" if ((y_pred > 0) == (expected_spin > 0)) else "ERR"
            print(f"  NAND({a},{b}) = {expected} (spin {expected_spin:+d}) | Pred: {y_pred:+.3f} | {ok}")
        
        pred_te2 = (Z_te2 > 0).float() * 2 - 1
        acc_te2 = (pred_te2 == Y_te).float().mean().item()
        print(f"\n  [GENERALISATION OOD] Precision NAND : {acc_te2*100:.1f}%")
    
    # --- TEST 3 : Penalty plus forte sur W_lin ---
    print("\n" + "=" * 70)
    print(" TEST 3 : Penalty renforcee sur W_lin (force le tenseur)")
    print("=" * 70)
    
    torch.manual_seed(seed)
    model3 = LIARNeuron(in_features=4, out_features=1, max_steps=5)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.05)
    
    for epoch in range(1200):
        optimizer3.zero_grad()
        output = model3(X_tr)
        # Penalite renforcee sur W_lin pour forcer le tenseur a dominer
        penalty = 0.001 * model3.get_structural_penalty() + 0.05 * torch.norm(model3.W_lin)
        loss = criterion(output, Y_tr) + penalty
        loss.backward()
        optimizer3.step()
    
    with torch.no_grad():
        Z_tr3 = model3(X_tr)
        pred_tr3 = (Z_tr3 > 0).float() * 2 - 1
        acc_tr3 = (pred_tr3 == Y_tr).float().mean().item()
        print(f"  Train Accuracy : {acc_tr3*100:.1f}%")
        
        Z_te3 = model3(X_te)
        for i, (a, b, expected) in enumerate(labels_test):
            y_pred = Z_te3[i].item()
            expected_spin = expected * 2 - 1
            ok = "OK" if ((y_pred > 0) == (expected_spin > 0)) else "ERR"
            print(f"  NAND({a},{b}) = {expected} (spin {expected_spin:+d}) | Pred: {y_pred:+.3f} | {ok}")
        
        pred_te3 = (Z_te3 > 0).float() * 2 - 1
        acc_te3 = (pred_te3 == Y_te).float().mean().item()
        print(f"\n  [GENERALISATION OOD] Precision NAND : {acc_te3*100:.1f}%")

if __name__ == '__main__':
    demo_spin_opcodes()
