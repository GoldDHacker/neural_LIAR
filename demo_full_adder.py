import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from liar_neuron import LIARNeuron


def _bit_to_spin(v: int) -> float:
    return float(v * 2 - 1)


def _inv_softplus(y: float) -> float:
    # Inverse numerique de softplus pour initialiser beta_raw.
    # softplus(x)=log(1+exp(x)) => x=log(exp(y)-1)
    y_t = torch.tensor(float(y))
    x = torch.log(torch.expm1(y_t))
    return float(x.item())


def _parse_float_list(s: str, expected_len: int):
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    vals = [float(p.replace(",", ".")) for p in parts]
    if len(vals) == 1 and expected_len > 1:
        return vals * expected_len
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} float(s) but got {len(vals)} from '{s}'")
    return vals


def train_full_adder():
    seed = int(os.environ.get("THERMO_DEMO_SEED", "42"))
    verbose = os.environ.get("THERMO_VERBOSE", "0").strip() not in ("", "0", "false", "False")

    log_every = int(os.environ.get("THERMO_LOG_EVERY_FULL_ADDER", "200"))

    binary_inputs = os.environ.get("THERMO_BINARY_INPUTS_FULL_ADDER", "0").strip() not in ("", "0", "false", "False")
    do_asserts = os.environ.get("THERMO_ASSERT_FULL_ADDER", "1").strip() not in ("", "0", "false", "False")

    best_ckpt = os.environ.get("THERMO_BEST_CHECKPOINT", "1").strip() not in ("", "0", "false", "False")
    early_stop = os.environ.get("THERMO_EARLY_STOP", "1").strip() not in ("", "0", "false", "False")

    lr_str = os.environ.get("THERMO_LR_FULL_ADDER", "0.03")
    lr = float(lr_str.replace(",", "."))

    struct_coeff_str = os.environ.get("THERMO_STRUCT_COEFF_FULL_ADDER", "0.0")
    struct_coeff = float(struct_coeff_str.replace(",", "."))

    epochs = int(os.environ.get("THERMO_EPOCHS_FULL_ADDER", "4000"))

    max_steps = int(os.environ.get("THERMO_MAX_STEPS_FULL_ADDER", "7"))
    full_tensor = os.environ.get("THERMO_FULL_TENSOR_FULL_ADDER", "0").strip() not in ("", "0", "false", "False")
    latent_dim = int(os.environ.get("THERMO_LATENT_DIM_FULL_ADDER", "16"))

    beta_init_str = os.environ.get("THERMO_BETA_INIT_FULL_ADDER", "")
    gate_init_str = os.environ.get("THERMO_GATE_INIT_FULL_ADDER", "")
    g3_init_str = os.environ.get("THERMO_G3_INIT_FULL_ADDER", "")
    freeze_gate = os.environ.get("THERMO_FREEZE_GATE_FULL_ADDER", "0").strip() not in ("", "0", "false", "False")
    freeze_beta = os.environ.get("THERMO_FREEZE_BETA_FULL_ADDER", "0").strip() not in ("", "0", "false", "False")

    beta_anneal = os.environ.get("THERMO_BETA_ANNEAL_FULL_ADDER", "1").strip() not in ("", "0", "false", "False")
    beta_start_str = os.environ.get("THERMO_BETA_START_FULL_ADDER", "0.4")
    beta_end_str = os.environ.get("THERMO_BETA_END_FULL_ADDER", "3.0")
    beta_frac_str = os.environ.get("THERMO_BETA_FRAC_FULL_ADDER", "0.7")
    beta_start = float(beta_start_str.replace(",", "."))
    beta_end = float(beta_end_str.replace(",", "."))
    beta_frac = float(beta_frac_str.replace(",", "."))

    two_phase = os.environ.get("THERMO_TWO_PHASE_FULL_ADDER", "1").strip() not in ("", "0", "false", "False")
    beta_train_str = os.environ.get("THERMO_BETA_TRAIN_FULL_ADDER", "0.3")
    beta_eval_str = os.environ.get("THERMO_BETA_EVAL_FULL_ADDER", "3.0")
    beta_train = float(beta_train_str.replace(",", "."))
    beta_eval = float(beta_eval_str.replace(",", "."))

    debug_fields = os.environ.get("THERMO_DEBUG_FIELDS_FULL_ADDER", "1").strip() not in ("", "0", "false", "False")

    disable_wave = os.environ.get("THERMO_DISABLE_WAVE_FULL_ADDER", "1").strip() not in ("", "0", "false", "False")

    torch.manual_seed(seed)

    X_list = []
    Y_list = []

    for a in [0, 1]:
        for b in [0, 1]:
            for cin in [0, 1]:
                s = (a ^ b) ^ cin
                cout = (a & b) | (cin & (a ^ b))

                if binary_inputs:
                    X_list.append([float(a), float(b), float(cin)])
                else:
                    X_list.append([_bit_to_spin(a), _bit_to_spin(b), _bit_to_spin(cin)])
                Y_list.append([_bit_to_spin(s), _bit_to_spin(cout)])

    X = torch.tensor(X_list)
    Y = torch.tensor(Y_list)

    model_latent_dim = None if full_tensor else latent_dim
    model = LIARNeuron(in_features=3, out_features=2, max_steps=max_steps, latent_dim=model_latent_dim)

    if disable_wave and hasattr(model, "wave_gate_raw"):
        with torch.no_grad():
            model.wave_gate_raw.data[:] = torch.tensor([-12.0] * model.out_features, dtype=model.wave_gate_raw.dtype)
        model.wave_gate_raw.requires_grad_(False)
        if hasattr(model, "phase_freq_base"):
            model.phase_freq_base.requires_grad_(False)
        if hasattr(model, "phase_freq_adapt"):
            model.phase_freq_adapt.requires_grad_(False)
        if hasattr(model, "phase_shift"):
            model.phase_shift.requires_grad_(False)
        if hasattr(model, "lambda_wave_raw"):
            model.lambda_wave_raw.requires_grad_(False)

    if beta_init_str.strip() not in ("", "none", "None"):
        beta_inits = _parse_float_list(beta_init_str, expected_len=model.out_features)
        with torch.no_grad():
            model.beta_raw.data[:] = torch.tensor([_inv_softplus(v) for v in beta_inits], dtype=model.beta_raw.dtype)

    if gate_init_str.strip() not in ("", "none", "None"):
        gate_inits = _parse_float_list(gate_init_str, expected_len=model.out_features)
        with torch.no_grad():
            model.tensor_gate_raw.data[:] = torch.tensor(gate_inits, dtype=model.tensor_gate_raw.dtype)

    if g3_init_str.strip() not in ("", "none", "None") and hasattr(model, "g3_raw"):
        g3_inits = _parse_float_list(g3_init_str, expected_len=model.out_features)
        with torch.no_grad():
            model.g3_raw.data[:] = torch.tensor(g3_inits, dtype=model.g3_raw.dtype)

    if freeze_gate:
        model.tensor_gate_raw.requires_grad_(False)
    if freeze_beta:
        model.beta_raw.requires_grad_(False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("==========================================================")
    print(" TEST STANDARD : Neurone L.I.A.R sur le Full-Adder")
    print("==========================================================")
    print(f" Seed = {seed}")
    print(f" LR = {lr}")
    print(f" Structural coeff = {struct_coeff}")
    print(f" binary_inputs = {bool(binary_inputs)}")
    print(f" full_tensor = {bool(full_tensor)}")
    print(f" latent_dim = {latent_dim}")
    print(f" max_steps = {max_steps}")
    if beta_init_str.strip() not in ("", "none", "None"):
        print(f" beta_init = {beta_init_str}")
    if gate_init_str.strip() not in ("", "none", "None"):
        print(f" gate_init_raw = {gate_init_str}")
    if g3_init_str.strip() not in ("", "none", "None"):
        print(f" g3_init_raw = {g3_init_str}")
    print(f" freeze_gate = {bool(freeze_gate)}")
    print(f" freeze_beta = {bool(freeze_beta)}")
    print(f" asserts = {bool(do_asserts)}")
    print(f" beta_anneal = {bool(beta_anneal)}")
    if beta_anneal and (not freeze_beta):
        print(f" beta_schedule = {beta_start} -> {beta_end} (frac={beta_frac})")
    print(f" two_phase = {bool(two_phase)}")
    if two_phase:
        print(f" beta_train = {beta_train} | beta_eval = {beta_eval}")
    print(f" debug_fields = {bool(debug_fields)}")
    print(f" disable_wave = {bool(disable_wave)}")
    print("Objectif : (A, B, Cin) -> (Sum, Cout) avec une seule couche")
    print("----------------------------------------------------------\n")

    best_state = None
    best_epoch = None
    best_loss = None

    def _meets_strict_success(output_tensor: torch.Tensor) -> bool:
        pred = (output_tensor > 0).float() * 2 - 1
        correct_now = (pred == Y).all().item()
        saturated_now = (output_tensor.abs() > 0.8).all().item()
        return bool(correct_now) and bool(saturated_now)

    for epoch in range(epochs):
        if two_phase:
            # Phase d'apprentissage : garder beta bas (chaud) pour conserver du gradient.
            with torch.no_grad():
                model.beta_raw.data[:] = torch.tensor(
                    [_inv_softplus(beta_train)] * model.out_features,
                    dtype=model.beta_raw.dtype,
                )
        if beta_anneal and (not freeze_beta):
            t = (epoch + 1) / max(1, epochs)
            t_norm = min(1.0, max(0.0, t / max(1e-8, beta_frac)))
            beta_now = beta_start + (beta_end - beta_start) * t_norm
            with torch.no_grad():
                model.beta_raw.data[:] = torch.tensor(
                    [_inv_softplus(beta_now)] * model.out_features,
                    dtype=model.beta_raw.dtype,
                )

        optimizer.zero_grad()

        output = model(X)
        mse = criterion(output, Y)
        structural = model.get_structural_penalty()
        wave_penalty = (
            model.get_wave_penalty()
            if (not disable_wave) and hasattr(model, "get_wave_penalty")
            else torch.tensor(0.0, device=mse.device)
        )
        loss = mse + struct_coeff * structural + wave_penalty

        if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
            print(f"[ERREUR NUMERIQUE] loss NaN/Inf detectee a l'epoch {epoch+1}")
            print(f"  mse = {mse.detach().item():.6f}")
            print(f"  structural = {structural.detach().item():.6f}")
            print(f"  beta = {model.beta.detach().cpu().numpy().tolist()}")
            print(f"  tensor_gate = {model.tensor_gate.detach().cpu().numpy().tolist()}")
            break

        loss.backward()
        optimizer.step()

        if log_every > 0 and ((epoch + 1) % log_every == 0):
            with torch.no_grad():
                pred_class = (output > 0).float() * 2 - 1
                acc = (pred_class == Y).float().mean().item()
                print(f"  Epoch {epoch+1:4d}/{epochs} | Loss = {loss.item():.6f} | Acc = {acc*100:5.1f}%")

        if verbose and ((epoch + 1) % 100 == 0 or epoch == 0):
            with torch.no_grad():
                pred_class = (output > 0).float() * 2 - 1
                acc = (pred_class == Y).float().mean().item()
                beta_vals = model.beta.detach().cpu().numpy().tolist()
                tg_vals = model.tensor_gate.detach().cpu().numpy().tolist()
                g1_vals = model.g1.detach().cpu().numpy().tolist() if hasattr(model, "g1") else None
                g2_vals = model.g2.detach().cpu().numpy().tolist() if hasattr(model, "g2") else None
                g3_vals = model.g3.detach().cpu().numpy().tolist() if hasattr(model, "g3") else None
                alphas = [float(a) for a in model.alpha.detach().cpu().numpy().tolist()]
                print(
                    f"  [VERBOSE:FULL_ADDER] ep={epoch+1:4d}/{epochs} | mse={mse.detach().item():.6f} | structural={structural.detach().item():.6f} | "
                    f"loss={loss.item():.6f} | acc={acc*100:5.1f}% | beta={beta_vals} | tensor_gate={tg_vals} | g1={g1_vals} | g2={g2_vals} | g3={g3_vals} | alpha={alphas}"
                )

        if best_ckpt and ((epoch + 1) % 10 == 0 or (epoch + 1) >= max(10, epochs - 100)):
            with torch.no_grad():
                if two_phase:
                    beta_raw_backup = model.beta_raw.detach().clone()
                    model.beta_raw.data[:] = torch.tensor(
                        [_inv_softplus(beta_eval)] * model.out_features,
                        dtype=model.beta_raw.dtype,
                    )
                    out_eval = model(X)
                    model.beta_raw.data[:] = beta_raw_backup
                    ok_now = _meets_strict_success(out_eval)
                else:
                    ok_now = _meets_strict_success(output)

                if ok_now:
                    loss_val = float(loss.item())
                    if best_loss is None or loss_val < best_loss:
                        best_loss = loss_val
                        best_epoch = int(epoch + 1)
                        best_state = copy.deepcopy(model.state_dict())

        if early_stop:
            with torch.no_grad():
                if two_phase:
                    beta_raw_backup = model.beta_raw.detach().clone()
                    model.beta_raw.data[:] = torch.tensor(
                        [_inv_softplus(beta_eval)] * model.out_features,
                        dtype=model.beta_raw.dtype,
                    )
                    out_eval = model(X)
                    model.beta_raw.data[:] = beta_raw_backup
                    ok_now = _meets_strict_success(out_eval)
                else:
                    ok_now = _meets_strict_success(output)

                if ok_now:
                    if not best_ckpt:
                        best_loss = float(loss.item())
                        best_epoch = int(epoch + 1)
                        best_state = copy.deepcopy(model.state_dict())
                    print(f"  [EARLY_STOP] condition atteinte a ep={epoch+1} (correct + sature).")
                    break

        if (epoch + 1) % 400 == 0:
            pass

    print("\n==========================================================")
    print(" VERIFICATION RIGOUREUSE DES CRITERES")
    print("==========================================================")

    with torch.no_grad():
        if best_state is not None:
            model.load_state_dict(best_state)

        if two_phase:
            # Phase d'evaluation : beta froid pour tester la saturation/commitment.
            model.beta_raw.data[:] = torch.tensor(
                [_inv_softplus(beta_eval)] * model.out_features,
                dtype=model.beta_raw.dtype,
            )

        final_out = model(X)
        pred_class = (final_out > 0).float() * 2 - 1

        acc = (pred_class == Y).float().mean().item()
        final_mse = criterion(final_out, Y).item()
        saturated = (final_out.abs() > 0.8).all().item()
        correct = (pred_class == Y).all().item()

        if best_ckpt:
            if best_state is None:
                print("\n  [BEST_CKPT] Aucun checkpoint valide n'a ete observe (correct + sature) pendant l'entrainement.")
            else:
                print("\n  [BEST_CKPT] Checkpoint retenu")
                print(f"    best_epoch = {best_epoch} | best_loss = {best_loss:.6f}")

        print("\n  (A, B, Cin) | Sum  Cout | Pred_S Pred_C | Status")
        print("  " + "-" * 52)
        all_ok = True
        for i in range(X.size(0)):
            a_s, b_s, c_s = [int(v.item()) for v in X[i]]
            s_true, c_true = int(Y[i, 0].item()), int(Y[i, 1].item())
            s_pred, c_pred = float(final_out[i, 0].item()), float(final_out[i, 1].item())

            ok = ((s_pred > 0) == (s_true > 0)) and ((c_pred > 0) == (c_true > 0))
            if not ok:
                all_ok = False
            status = "OK" if ok else "FAIL"
            print(f"  ({a_s:+d},{b_s:+d},{c_s:+d})   | {s_true:+d}   {c_true:+d}  | {s_pred:+.3f} {c_pred:+.3f} | {status}")

        print("  " + "-" * 52)
        print(f"  Accuracy globale = {acc*100:.1f}% | MSE = {final_mse:.6f} | Satur1 = {bool(saturated)}")

        strict_success = bool(correct) and bool(saturated) and (final_mse < 0.05)

        if all_ok and strict_success:
            print("\n  -> [SUCCES] Le neurone a appris le Full-Adder (Sum et Cout simultanes)")
        else:
            print("\n  -> [ECHEC] Le neurone a echoue a apprendre le Full-Adder")

        if debug_fields:
            x_log = model.perception(X)
            e_lin = F.linear(x_log, model.W_lin, model.b)
            if getattr(model, "latent_dim", None) is None:
                e_tensor = torch.einsum('bi,oij,bj->bo', x_log, model.W_tensor, x_log)
            else:
                q = torch.einsum('bi,oid->bod', x_log, model.U)
                k = torch.einsum('bi,oid->bod', x_log, model.V)
                e_tensor = torch.sum(q * k * model.S_raw.unsqueeze(0), dim=-1)

            if hasattr(model, "A3") and hasattr(model, "g3"):
                a3 = torch.einsum('bi,oid->bod', x_log, model.A3)
                b3 = torch.einsum('bi,oid->bod', x_log, model.B3)
                c3 = torch.einsum('bi,oid->bod', x_log, model.C3)
                e_tri = torch.sum(a3 * b3 * c3 * model.S3_raw.unsqueeze(0), dim=-1)
                e_total = model.g1.unsqueeze(0) * e_lin + model.g2.unsqueeze(0) * e_tensor + model.g3.unsqueeze(0) * e_tri
            else:
                e_total = e_lin + model.tensor_gate.unsqueeze(0) * e_tensor

            mismatch = (pred_class != Y)
            bad_rows = mismatch.any(dim=1)
            n_bad = int(bad_rows.sum().item())
            if n_bad > 0:
                print(f"\n  [DEBUG_FIELDS] n_bad={n_bad}/8 (E_lin/E_tensor/E_total)")
                for i in range(X.size(0)):
                    if not bool(bad_rows[i].item()):
                        continue
                    x_i = X[i].tolist()
                    y_i = Y[i].tolist()
                    pred_i = final_out[i].tolist()
                    e_lin_i = e_lin[i].tolist()
                    e_tensor_i = e_tensor[i].tolist()
                    e_total_i = e_total[i].tolist()
                    print(f"    idx={i} | x={x_i} | y={y_i} | pred={pred_i}")
                    print(f"      E_lin={e_lin_i}")
                    print(f"      E_tensor={e_tensor_i}")
                    print(f"      gate={model.tensor_gate.detach().cpu().numpy().tolist()} | beta={model.beta.detach().cpu().numpy().tolist()}")
                    print(f"      E_total={e_total_i}")

        print(f"\n  Facteur Thermique (Beta) : {model.beta.detach().cpu().numpy().tolist()}")
        tg = model.tensor_gate.detach().cpu().numpy().tolist()
        print(f"  Gate Tensorielle (tensor_gate) : {tg}")

        if do_asserts:
            assert all_ok, "Echec FULL_ADDER_GROWTH"
            assert saturated and final_mse < 0.05, "Echec FULL_ADDER_STRICT"


if __name__ == '__main__':
    train_full_adder()
