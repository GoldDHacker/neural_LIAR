import os
import torch
import torch.nn as nn
import torch.optim as optim
from liar_neuron import LIARNeuron


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    y = torch.clamp(y, min=1e-6)
    return torch.log(torch.expm1(y))


def _beta_schedule(epoch: int, total_epochs: int, beta_start: float, beta_end: float) -> float:
    if total_epochs <= 1:
        return float(beta_end)
    t = float(epoch) / float(total_epochs - 1)
    return float(beta_start + (beta_end - beta_start) * t)

def train_alu():
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    verbose = os.environ.get("THERMO_VERBOSE", "0").strip() not in ("", "0", "false", "False")
    dump_history = os.environ.get("THERMO_DUMP_HISTORY", "0").strip() not in ("", "0", "false", "False")
    dump_index = int(os.environ.get("THERMO_DUMP_INDEX", "15"))
    lr_alu_str = os.environ.get("THERMO_LR_ALU", "0.02")
    lr_alu = float(lr_alu_str.replace(",", "."))
    struct_coeff_str = os.environ.get("THERMO_STRUCT_COEFF_ALU", "0.001")
    struct_coeff = float(struct_coeff_str.replace(",", "."))
    epochs = int(os.environ.get("THERMO_EPOCHS_ALU", "2000"))

    beta_adversarial = os.environ.get("THERMO_BETA_ADVERSARIAL", "0").strip() not in ("", "0", "false", "False")
    beta_init_str = os.environ.get("THERMO_BETA_INIT_ALU", "5.0")
    beta_init = float(beta_init_str.replace(",", "."))

    beta_anneal = os.environ.get("THERMO_BETA_ANNEAL", "0").strip() not in ("", "0", "false", "False")
    beta_anneal_start_str = os.environ.get("THERMO_BETA_ANNEAL_START", "0.25")
    beta_anneal_end_str = os.environ.get("THERMO_BETA_ANNEAL_END", "2.0")
    beta_anneal_start = float(beta_anneal_start_str.replace(",", "."))
    beta_anneal_end = float(beta_anneal_end_str.replace(",", "."))
    beta_anneal_epochs = int(os.environ.get("THERMO_BETA_ANNEAL_EPOCHS", str(epochs)))
    beta_learn_after_anneal = os.environ.get("THERMO_BETA_LEARN_AFTER_ANNEAL", "0").strip() not in ("", "0", "false", "False")

    log_csv_path = os.environ.get("THERMO_LOG_CSV", "").strip()
    log_every = int(os.environ.get("THERMO_LOG_EVERY", "50"))
    csv_file = None
    if log_csv_path:
        csv_file = open(log_csv_path, "w", encoding="utf-8")
        csv_file.write("epoch,mse,loss,acc,beta,tensor_gate\n")
    torch.manual_seed(seed)
    
    # Dataset ALU (Arithmetic Logic Unit)
    # Entrees : O1, O2 (Opcode / Routing), A, B (Operandes)
    # Total : 16 echantillons de 4 bits
    
    # Opcode mapping:
    # 00 : AND
    # 01 : OR
    # 10 : XOR
    # 11 : NAND
    
    X_list = []
    Y_list = []
    
    # Generer la table de hachage complete
    for o1 in [0, 1]:
        for o2 in [0, 1]:
            for a in [0, 1]:
                for b in [0, 1]:
                    if o1 == 0 and o2 == 0:     # AND
                        res = a & b
                    elif o1 == 0 and o2 == 1:   # OR
                        res = a | b
                    elif o1 == 1 and o2 == 0:   # XOR
                        res = a ^ b
                    elif o1 == 1 and o2 == 1:   # NAND
                        res = 1 - (a & b)
                        
                    X_list.append([float(o1), float(o2), float(a), float(b)])
                    # Mapping de la cible vers [-1, 1]
                    Y_list.append([float(res * 2 - 1)])
                    
    X = torch.tensor(X_list)
    Y = torch.tensor(Y_list)
    
    # Modele : 4 entrees (2 opcode, 2 data), 1 sortie.
    model = LIARNeuron(in_features=4, out_features=1, max_steps=5)

    if beta_adversarial:
        with torch.no_grad():
            init_raw = _inv_softplus(torch.tensor([beta_init], dtype=model.beta_raw.dtype, device=model.beta_raw.device))
            model.beta_raw.copy_(init_raw)

    if beta_anneal:
        model.beta_raw.requires_grad_(False)
    
    # ALU est beaucoup plus complexe, nous laissons l'Adam travailler plus longtemps
    optimizer = optim.Adam(model.parameters(), lr=lr_alu)
    criterion = nn.MSELoss()
    
    print("==========================================================")
    print(" STRESS-TEST ULTIME : Neurone L.I.A.R sur l'A.L.U.")
    print("==========================================================")
    print(f" Seed = {seed}")
    print(f" LR = {lr_alu}")
    print(f" Structural coeff = {struct_coeff}")
    if beta_adversarial:
        print(f" [ADVERSARIAL_INIT] beta_init = {beta_init}")
    if beta_anneal:
        print(
            f" [BETA_ANNEAL] start={beta_anneal_start} end={beta_anneal_end} anneal_epochs={beta_anneal_epochs} | "
            f"beta_learnable={bool(model.beta_raw.requires_grad)}"
        )
    print("Objectif : Une seule couche doit dynamiquement commuter entre")
    print("AND, OR, XOR et NAND en fonction des bits de routage virtuels.")
    print("----------------------------------------------------------\n")

    for epoch in range(epochs):
        if beta_anneal:
            e = epoch if epoch < beta_anneal_epochs else (beta_anneal_epochs - 1)
            beta_t = _beta_schedule(e, beta_anneal_epochs, beta_anneal_start, beta_anneal_end)
            with torch.no_grad():
                raw_t = _inv_softplus(torch.tensor([beta_t], dtype=model.beta_raw.dtype, device=model.beta_raw.device))
                model.beta_raw.copy_(raw_t)

            if beta_learn_after_anneal and epoch + 1 == beta_anneal_epochs:
                model.beta_raw.requires_grad_(True)

        optimizer.zero_grad()
        
        output = model(X)
        
        mse_loss = criterion(output, Y)
        # Trop de penalite structurelle sur un circuit de routage complexe peut empecher la solution
        structural = model.get_structural_penalty()
        wave_penalty = model.get_wave_penalty() if hasattr(model, "get_wave_penalty") else torch.tensor(0.0, device=mse_loss.device)
        loss = mse_loss + struct_coeff * structural + wave_penalty

        if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
            print(f"[ERREUR NUMERIQUE] loss NaN/Inf detectee a l'epoch {epoch+1}")
            print(f"  mse_loss = {mse_loss.item():.6f}")
            print(f"  structural_penalty = {model.get_structural_penalty().detach().item():.6f}")
            print(f"  beta = {model.beta.detach().item():.6f}")
            alphas = [f"{a:.6f}" for a in model.alpha.detach().cpu().numpy()]
            print(f"  alpha = {alphas}")
            print(f"  tensor_gate = {model.tensor_gate.detach().item():.6f}")
            break
        
        loss.backward()
        optimizer.step()

        if csv_file is not None and ((epoch + 1) % log_every == 0 or epoch == 0 or (epoch + 1) == epochs):
            with torch.no_grad():
                pred_class = (output > 0).float() * 2 - 1
                acc = (pred_class == Y).float().mean().item()
                beta_val = model.beta.detach().item()
                tg = model.tensor_gate.detach().item()
                csv_file.write(
                    f"{epoch+1},{mse_loss.item():.12f},{loss.item():.12f},{acc:.6f},{beta_val:.8f},{tg:.8f}\n"
                )
        
        if (epoch + 1) % 200 == 0:
            pred_class = (output > 0).float() * 2 - 1
            acc = (pred_class == Y).float().mean().item()
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss = {loss.item():.5f} | Acc globale = {acc*100:3.0f}%")

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            with torch.no_grad():
                pred_class = (output > 0).float() * 2 - 1
                acc = (pred_class == Y).float().mean().item()
                structural_val = structural.detach().item()
                beta_val = model.beta.detach().item()
                tg = model.tensor_gate.detach().item()
                alphas = [float(a) for a in model.alpha.detach().cpu().numpy().tolist()]
                print(
                    f"[VERBOSE] ep={epoch+1:4d} | mse={mse_loss.item():.6f} | structural={structural_val:.6f} | "
                    f"loss={loss.item():.6f} | acc={acc*100:5.1f}% | beta={beta_val:.4f} | tensor_gate={tg:.4f} | alpha={alphas}"
                )

    print("\n==========================================================")
    print(" VERIFICATION RIGOUREUSE DES CRITERES (COMME VERSION A/B)")
    print("==========================================================")
    
    with torch.no_grad():
        final_out, _ = model(X, return_history=True)
        acc = (((final_out > 0).float() * 2 - 1) == Y).float().mean().item()
        final_mse = criterion(final_out, Y).item()

        if dump_history:
            if dump_index < 0 or dump_index >= X.size(0):
                print(f"\n[HISTORY_DUMP] Index invalide: dump_index={dump_index} (doit etre dans [0, {X.size(0)-1}])")
            else:
                out_full, history = model(X, return_history=True)
                x_i = X[dump_index]
                y_true_i = int(Y[dump_index][0].item())
                y_pred_i = float(out_full[dump_index][0].item())
                y_cls_i = int((((out_full[dump_index][0] > 0).float() * 2 - 1)).item())
                o_str = f"{int(x_i[0].item())}{int(x_i[1].item())}"
                ab_str = f"{int(x_i[2].item())},{int(x_i[3].item())}"
                print("\n[HISTORY_DUMP] Trajectoire interne de l'attracteur (Ising)")
                print(f"  idx={dump_index} | opcode={o_str} | (A,B)={ab_str} | y_true={y_true_i:+d} | y_cls={y_cls_i:+d} | y_pred={y_pred_i:+.6f}")
                print(f"  max_steps = {model.max_steps}")
                for t, z_t in enumerate(history, start=1):
                    z_val = float(z_t[dump_index][0].item())
                    print(f"    step {t}: Z = {z_val:+.6f}")
        
    print("\n[TEST L.I.A.R : ALU_GROWTH (Capacite de Multiplexage Dynamique)]")
    print(f"-> Precision atteinte : {acc*100:3.0f}%")
    if acc == 1.0:
        print("-> [SUCCES] L'architecture s'est pliee en 4 dimensions simultanement !")
    else:
        print("-> [ECHEC] Le reseau n'a pas la capacite de router 4 comportements distincts.")

    print("\n[TEST L.I.A.R : ALU_STRICT (Robustesse Thermodynamique Globale)]")
    all_strict = True
    
    print(f"{'Opcode':^6} | {'(A, B)':^6} | {'Truth':^5} | {'Pred':^6} | {'Status':^12}")
    print("-" * 50)
    
    for i in range(16):
        o_str = f"{int(X[i][0])}{int(X[i][1])}"
        ab_str = f"{int(X[i][2])},{int(X[i][3])}"
        y_true = int(Y[i][0].item())
        y_pred = final_out[i][0].item()
        
        is_strict = abs(y_pred) >= 0.8
        if not is_strict:
            all_strict = False
        
        status = "OK (SAT)" if is_strict else "FAIL (INC)"
        # Ne printer que quelques lignes ou les erreurs s'il y en a beaucoup
        if verbose or i % 4 == 0 or not is_strict:
            print(f"{o_str:^6} | {ab_str:^6} | {y_true:+4d} | {y_pred:+6.3f} | {status}")
        elif i % 4 == 3:
            print(f"  ... (+3 autres echantillons {o_str})")
        
    print("-" * 50)
    print(f"-> MSE Finale : {final_mse:.5f}")
    if all_strict and final_mse < 0.05:
        print("-> [SUCCES] Saturation parfaite sur l'ensemble de l'Hypercube 4D.")
    else:
        print("-> [ECHEC] Incoherence magnetique detectee. L'attracteur a fondu.")

    if verbose or acc < 1.0 or not (all_strict and final_mse < 0.05):
        pred_class = (final_out > 0).float() * 2 - 1
        mismatch = (pred_class != Y).squeeze(1)
        n_bad = int(mismatch.sum().item())
        print(f"\n[DETAIL ERREURS] n_bad = {n_bad}/16")
        if n_bad > 0:
            for i in range(16):
                if not bool(mismatch[i].item()):
                    continue
                o_str = f"{int(X[i][0])}{int(X[i][1])}"
                ab_str = f"{int(X[i][2])},{int(X[i][3])}"
                y_true = int(Y[i][0].item())
                y_pred = final_out[i][0].item()
                y_cls = int(pred_class[i][0].item())
                print(f"  idx={i:2d} | opcode={o_str} | (A,B)={ab_str} | y_true={y_true:+d} | y_cls={y_cls:+d} | y_pred={y_pred:+.6f}")

    print(f"\n  Facteur Thermique (Beta) : {model.beta.detach().item():.3f}")
    alphas = [f"{a:.3f}" for a in model.alpha.detach().numpy()]
    print(f"  Puissances de Perception (Alpha par input) : {alphas}")

    if csv_file is not None:
        csv_file.close()

if __name__ == '__main__':
    train_alu()
