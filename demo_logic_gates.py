"""
L.I.A.R : Tests de Capacite Logique Complets
=============================================
Verifie que le Neurone Thermodynamique (Systeme 1) 
peut apprendre TOUTES les fonctions logiques fondamentales :

1. Les 16 Fonctions Booleennes a 2 entrees 
   (FALSE, AND, A>B, A, B>A, B, XOR, OR, NOR, XNOR, NOT_B, A>=B, NOT_A, B>=A, NAND, TRUE)
2. Le Demi-Additionneur (Half-Adder) : Sortie 2 bits (Sum=XOR, Carry=AND)
3. Le Comparateur (A > B) : Fonction non-symetrique

Encodage : Entrees/Sorties en {-1, +1} (spins d'Ising)
"""
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from liar_neuron import LIARNeuron

# =====================================================================
# DEFINITION DES 16 FONCTIONS BOOLEENNES A 2 ENTREES
# =====================================================================
# Entrees : (A, B) en {-1, +1}
# Sorties : f(A, B) en {-1, +1}
# Ordre des entrees : (-1,-1), (-1,+1), (+1,-1), (+1,+1)

BOOL_FUNCTIONS = {
    "FALSE":   [-1, -1, -1, -1],
    "AND":     [-1, -1, -1, +1],
    "A>B":     [-1, -1, +1, -1],  # A AND NOT B
    "A":       [-1, -1, +1, +1],
    "B>A":     [-1, +1, -1, -1],  # B AND NOT A
    "B":       [-1, +1, -1, +1],
    "XOR":     [-1, +1, +1, -1],
    "OR":      [-1, +1, +1, +1],
    "NOR":     [+1, -1, -1, -1],
    "XNOR":    [+1, -1, -1, +1],
    "NOT_B":   [+1, -1, +1, -1],
    "A>=B":    [+1, -1, +1, +1],  # A OR NOT B (implication B -> A)
    "NOT_A":   [+1, +1, -1, -1],
    "B>=A":    [+1, +1, -1, +1],  # B OR NOT A (implication A -> B)
    "NAND":    [+1, +1, +1, -1],
    "TRUE":    [+1, +1, +1, +1],
}

X_2BIT = torch.tensor([
    [-1.0, -1.0],
    [-1.0, +1.0],
    [+1.0, -1.0],
    [+1.0, +1.0],
])

def test_all_16_boolean():
    """Test 1 : Apprendre les 16 fonctions booleennes individuellement."""
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    verbose = os.environ.get("THERMO_VERBOSE", "0").strip() not in ("", "0", "false", "False")
    print("=" * 65)
    print(" L.I.A.R : Test des 16 Fonctions Booleennes a 2 Entrees")
    print("=" * 65)
    print(f" Seed = {seed}")
    
    results = {}
    
    for name, truth_table in BOOL_FUNCTIONS.items():
        torch.manual_seed(seed)
        Y = torch.tensor(truth_table, dtype=torch.float32).unsqueeze(1)
        
        model = LIARNeuron(in_features=2, out_features=1, max_steps=5, latent_dim=4)
        optimizer = optim.Adam(model.parameters(), lr=0.08)
        criterion = nn.MSELoss()
        
        for epoch in range(500):
            optimizer.zero_grad()
            output = model(X_2BIT)
            mse = criterion(output, Y)
            structural = model.get_structural_penalty()
            wave_penalty = model.get_wave_penalty() if hasattr(model, "get_wave_penalty") else torch.tensor(0.0, device=mse.device)
            loss = mse + 0.001 * structural + wave_penalty
            loss.backward()
            optimizer.step()

            if verbose and ((epoch + 1) % 100 == 0 or epoch == 0):
                with torch.no_grad():
                    pred_class = (output > 0).float() * 2 - 1
                    acc = (pred_class == Y).float().mean().item()
                    structural = model.get_structural_penalty().detach().item()
                    beta_val = model.beta.detach().item()
                    tg = model.tensor_gate.detach().item()
                    alphas = [float(a) for a in model.alpha.detach().cpu().numpy().tolist()]
                    mse = criterion(output, Y).detach().item()
                    print(
                        f"  [VERBOSE:{name}] ep={epoch+1:3d}/500 | mse={mse:.6f} | structural={structural:.6f} | "
                        f"loss={loss.item():.6f} | acc={acc*100:5.1f}% | beta={beta_val:.4f} | tensor_gate={tg:.4f} | alpha={alphas}"
                    )
        
        with torch.no_grad():
            final_out = model(X_2BIT)
            pred = ((final_out > 0).float() * 2 - 1).squeeze()
            target = torch.tensor(truth_table, dtype=torch.float32)
            correct = (pred == target).all().item()
            saturated = (final_out.abs() > 0.8).all().item()
            
        status = "OK" if correct and saturated else ("~OK" if correct else "FAIL")
        results[name] = status
        
        vals = " ".join([f"{v:+.2f}" for v in final_out.squeeze().tolist()])
        print(f"  {name:7s} : [{vals}] | {status}")
    
    success = sum(1 for v in results.values() if v in ("OK", "~OK"))
    print(f"\n  BILAN : {success}/16 fonctions apprises")
    print("-" * 65)
    return results

def test_half_adder():
    """Test 2 : Demi-Additionneur (Sum=XOR, Carry=AND) -> Sortie 2 bits."""
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    verbose = os.environ.get("THERMO_VERBOSE", "0").strip() not in ("", "0", "false", "False")
    best_ckpt = os.environ.get("THERMO_BEST_CHECKPOINT", "0").strip() not in ("", "0", "false", "False")
    early_stop = os.environ.get("THERMO_EARLY_STOP", "1").strip() not in ("", "0", "false", "False")
    lr_half_adder_str = os.environ.get("THERMO_LR_HALF_ADDER", "0.05")
    lr_half_adder = float(lr_half_adder_str.replace(",", "."))
    print("\n" + "=" * 65)
    print(" L.I.A.R : Test du Demi-Additionneur (Half-Adder)")
    print("=" * 65)
    print(f" Seed = {seed}")
    print(f" LR = {lr_half_adder}")
    print(" Sortie attendue : [Sum=XOR, Carry=AND]")
    print("-" * 65)
    
    torch.manual_seed(seed)
    
    # Verite : Sum=XOR, Carry=AND
    # (-1,-1) -> Sum=-1, Carry=-1
    # (-1,+1) -> Sum=+1, Carry=-1
    # (+1,-1) -> Sum=+1, Carry=-1
    # (+1,+1) -> Sum=-1, Carry=+1
    Y = torch.tensor([
        [-1.0, -1.0],
        [+1.0, -1.0],
        [+1.0, -1.0],
        [-1.0, +1.0],
    ])
    
    # UN SEUL neurone avec 2 sorties !
    model = LIARNeuron(in_features=2, out_features=2, max_steps=5, latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=lr_half_adder)
    criterion = nn.MSELoss()

    last_loss_val = None
    best_state = None
    best_epoch = None
    best_loss = None
    
    for epoch in range(800):
        optimizer.zero_grad()
        output = model(X_2BIT)
        mse = criterion(output, Y)
        structural = model.get_structural_penalty()
        loss = mse + 0.001 * structural
        loss.backward()
        optimizer.step()

        if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
            print(f"  [ERREUR NUMERIQUE] loss NaN/Inf detectee a l'epoch {epoch+1}")
            print(f"    mse = {mse.detach().item():.6f} | structural = {structural.detach().item():.6f}")
            beta_vals = model.beta.detach().cpu().numpy().tolist()
            tg_vals = model.tensor_gate.detach().cpu().numpy().tolist()
            print(f"    beta = {beta_vals} | tensor_gate = {tg_vals}")
            alphas = [f"{a:.6f}" for a in model.alpha.detach().cpu().numpy()]
            print(f"    alpha = {alphas}")
            break

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            with torch.no_grad():
                pred_class = (output > 0).float() * 2 - 1
                acc = (pred_class == Y).float().mean().item()
                beta_vals = model.beta.detach().cpu().numpy().tolist()
                tg_vals = model.tensor_gate.detach().cpu().numpy().tolist()
                alphas = [float(a) for a in model.alpha.detach().cpu().numpy().tolist()]
                print(
                    f"  [VERBOSE:HALF_ADDER] ep={epoch+1:4d}/800 | mse={mse.detach().item():.6f} | structural={structural.detach().item():.6f} | "
                    f"loss={loss.item():.6f} | acc={acc*100:5.1f}% | beta={beta_vals} | tensor_gate={tg_vals} | alpha={alphas}"
                )

        if best_ckpt and ((epoch + 1) % 5 == 0 or (epoch + 1) >= 780):
            with torch.no_grad():
                pred_class = (output > 0).float() * 2 - 1
                correct_now = (pred_class == Y).all().item()
                saturated_now = (output.abs() > 0.8).all().item()
                if correct_now and saturated_now:
                    loss_val = float(loss.item())
                    if best_loss is None or loss_val < best_loss:
                        best_loss = loss_val
                        best_epoch = int(epoch + 1)
                        best_state = copy.deepcopy(model.state_dict())

        if early_stop:
            with torch.no_grad():
                pred_class = (output > 0).float() * 2 - 1
                correct_now = (pred_class == Y).all().item()
                saturated_now = (output.abs() > 0.8).all().item()
                if correct_now and saturated_now:
                    if not best_ckpt:
                        best_loss = float(loss.item())
                        best_epoch = int(epoch + 1)
                        best_state = copy.deepcopy(model.state_dict())
                    print(f"  [EARLY_STOP] condition atteinte a ep={epoch+1} (correct + sature).")
                    break

        if last_loss_val is not None:
            loss_val = float(loss.item())
            if loss_val > last_loss_val * 5.0 and (epoch + 1) >= 400:
                print(f"  [ALERTE] saut de loss detecte a ep={epoch+1} : {last_loss_val:.6f} -> {loss_val:.6f}")
                beta_vals = model.beta.detach().cpu().numpy().tolist()
                tg_vals = model.tensor_gate.detach().cpu().numpy().tolist()
                print(f"    beta = {beta_vals} | tensor_gate = {tg_vals}")
        last_loss_val = float(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1:4d}/800 | Loss = {loss.item():.5f}")
    
    with torch.no_grad():
        if best_state is not None:
            model.load_state_dict(best_state)
        final_out = model(X_2BIT)
        pred = (final_out > 0).float() * 2 - 1
        correct = (pred == Y).all().item()

        if best_ckpt:
            if best_state is None:
                print("\n  [BEST_CKPT] Aucun checkpoint valide n'a ete observe (correct + sature) pendant l'entrainement.")
            else:
                model_best = LIARNeuron(in_features=2, out_features=2, max_steps=5, latent_dim=4)
                model_best.load_state_dict(best_state)
                best_out = model_best(X_2BIT)
                best_pred = (best_out > 0).float() * 2 - 1
                best_correct = (best_pred == Y).all().item()
                best_saturated = (best_out.abs() > 0.8).all().item()
                best_mse = criterion(best_out, Y).item()
                last_mse = criterion(final_out, Y).item()
                print("\n  [BEST_CKPT] Comparaison best vs last")
                print(f"    best_epoch = {best_epoch} | best_loss = {best_loss:.6f} | best_mse = {best_mse:.6f} | best_correct = {best_correct} | best_saturated = {best_saturated}")
                print(f"    last_epoch = 800 | last_mse = {last_mse:.6f} | last_correct = {bool(correct)}")
        
        print(f"\n  {'Entree':12s} | {'Sum':>6s} {'Carry':>6s} | {'Pred_S':>6s} {'Pred_C':>6s} | Status")
        print("  " + "-" * 55)
        all_ok = True
        for i in range(4):
            a, b = int(X_2BIT[i, 0].item()), int(X_2BIT[i, 1].item())
            s_true, c_true = int(Y[i, 0].item()), int(Y[i, 1].item())
            s_pred, c_pred = final_out[i, 0].item(), final_out[i, 1].item()
            ok = (s_pred > 0) == (s_true > 0) and (c_pred > 0) == (c_true > 0)
            if not ok: all_ok = False
            status = "OK" if ok else "FAIL"
            print(f"  ({a:+d}, {b:+d})     | {s_true:+d}     {c_true:+d}     | {s_pred:+.3f}  {c_pred:+.3f}  | {status}")
    
    result = "SUCCES" if all_ok else "ECHEC"
    print(f"\n  -> [{result}] Le neurone a {'appris' if all_ok else 'echoue a apprendre'} le Half-Adder (2 fonctions simultanées)")
    print("-" * 65)
    return all_ok

def test_comparator():
    """Test 3 : Comparateur A > B (fonction non-symetrique)."""
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    print("\n" + "=" * 65)
    print(" L.I.A.R : Test du Comparateur (A > B)")
    print("=" * 65)
    print(f" Seed = {seed}")
    print(" Fonction non-symetrique : seul (+1, -1) donne +1")
    print("-" * 65)
    
    torch.manual_seed(seed)
    
    # A > B : Vrai uniquement si A=+1 et B=-1
    Y = torch.tensor([[-1.0], [-1.0], [+1.0], [-1.0]])
    
    model = LIARNeuron(in_features=2, out_features=1, max_steps=5, latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.08)
    criterion = nn.MSELoss()
    
    for epoch in range(500):
        optimizer.zero_grad()
        output = model(X_2BIT)
        loss = criterion(output, Y) + 0.001 * model.get_structural_penalty()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_out = model(X_2BIT)
        pred = (final_out > 0).float() * 2 - 1
        
        all_ok = True
        for i in range(4):
            a, b = int(X_2BIT[i, 0].item()), int(X_2BIT[i, 1].item())
            expected = int(Y[i, 0].item())
            got = final_out[i, 0].item()
            ok = (got > 0) == (expected > 0)
            if not ok: all_ok = False
            status = "OK" if ok else "FAIL"
            sym = ">" if expected > 0 else "<=" 
            print(f"  A={a:+d}, B={b:+d} -> A{sym}B | Attendu: {expected:+d} | Predit: {got:+.3f} | {status}")
        
    result = "SUCCES" if all_ok else "ECHEC"
    print(f"\n  -> [{result}] Le neurone {'comprend' if all_ok else 'ne comprend pas'} l'asymetrie de l'ordre (A > B)")
    
    # Bonus : verifier la saturation
    sat = (final_out.abs() > 0.8).all().item()
    if sat:
        print("  -> [SATURE] Les attracteurs sont parfaitement bifurques.")
    print("-" * 65)
    return all_ok

if __name__ == '__main__':
    print()
    r16 = test_all_16_boolean()
    ha = test_half_adder()
    comp = test_comparator()
    
    print("\n" + "=" * 65)
    print(" BILAN GLOBAL DE CAPACITE LOGIQUE (L.I.A.R)")
    print("=" * 65)
    n16 = sum(1 for v in r16.values() if v in ("OK", "~OK"))
    print(f"  16 Fonctions Booleennes : {n16}/16")
    print(f"  Demi-Additionneur       : {'OK' if ha else 'FAIL'}")
    print(f"  Comparateur A > B       : {'OK' if comp else 'FAIL'}")
    print("=" * 65)
