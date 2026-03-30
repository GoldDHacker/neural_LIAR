import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from liar_neuron import LIARNeuron

class DeepMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

class SimpleGRU(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gru = nn.GRU(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        x_seq = x.unsqueeze(-1)
        _, h_n = self.gru(x_seq)
        return self.fc(h_n[-1])

def generate_parity_data(batch_size, n_bits, device):
    X = torch.randint(0, 2, (batch_size, n_bits), device=device).float() * 2 - 1
    Y = torch.prod(X, dim=1, keepdim=True)
    return X, Y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(
    model_name,
    n_bits,
    seed,
    epochs=1000,
    batch_size=256,
    lr=0.005,
    steps=5,
    latent_dim=None,
    struct_coeff=0.001,
    disable_wave=False,
    device="cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if model_name == "LIAR":
        eff_latent_dim = latent_dim
        if eff_latent_dim is None:
            eff_latent_dim = min(16, n_bits // 2)
        model = LIARNeuron(
            in_features=n_bits,
            out_features=1,
            max_steps=steps,
            latent_dim=eff_latent_dim,
            max_order=3,
        ).to(device)
    elif model_name == "MLP":
        model = DeepMLP(in_features=n_bits).to(device)
    elif model_name == "GRU":
        model = SimpleGRU(in_features=n_bits).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    num_params = count_parameters(model)
    model.train()
    train_t0 = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        X, Y = generate_parity_data(batch_size, n_bits, device)
        
        if model_name == "LIAR":
            Z_out = model(X)
            out = Z_out[:, 0:1]
            loss = criterion(out, Y) + struct_coeff * model.get_structural_penalty()
            if (not disable_wave) and hasattr(model, 'get_wave_penalty'):
                loss += struct_coeff * model.get_wave_penalty()
        else:
            out = model(X)
            loss = criterion(out, Y)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    train_seconds = time.time() - train_t0
        
    model.eval()
    with torch.no_grad():
        X_test, Y_test = generate_parity_data(2000, n_bits, device)
        if model_name == "LIAR":
            Z_out = model(X_test)
            out_test = Z_out[:, 0:1]
        else:
            out_test = model(X_test)
            
        preds = torch.sign(out_test)
        correct = (preds == Y_test).float().sum().item()
        accuracy = (correct / 2000.0) * 100
        
    return accuracy, num_params, train_seconds

def run_comparison_benchmark(num_seeds, epochs=1000, batch_size=256, lr=0.005, steps=5, latent_dim=None, struct_coeff=0.001, disable_wave=False):
    print(f"\n{'='*80}")
    print(f" BENCHMARK DE COMPARAISON : L.I.A.R vs MLP vs GRU sur Parite Globale")
    print(f" Nombres de Seeds par Dimension : {num_seeds}")
    print(f"{'='*80}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Appareil d'execution : {device}")
    
    N_BITS_LIST = [4, 8, 16, 32, 64]
    MODELS = ["LIAR", "MLP", "GRU"]
    
    for n_bits in N_BITS_LIST:
        print(f"\n[ Test N={n_bits} Bits | Entrainement sur {num_seeds} seeds ]")
        print("-" * 50)
        
        params_info = {}
        for model_name in MODELS:
            accuracies = []
            times_s = []
            for seed in range(num_seeds):
                acc, params, train_seconds = train_model(
                    model_name,
                    n_bits,
                    seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    steps=steps,
                    latent_dim=latent_dim,
                    struct_coeff=struct_coeff,
                    disable_wave=disable_wave,
                    device=device,
                )
                accuracies.append(acc)
                times_s.append(train_seconds)
                params_info[model_name] = params
            
            acc_array = np.array(accuracies)
            mean_acc = acc_array.mean()
            std_acc = acc_array.std()
            success_rate = (acc_array > 90.0).mean() * 100
            times_array = np.array(times_s)
            mean_time = times_array.mean()
            std_time = times_array.std()
            
            print(f" Modele {model_name:<5} | Params : {params_info[model_name]:>6d} | Moyenne : {mean_acc:>5.1f}% (+-{std_acc:>4.1f}%) | Succes : {success_rate:>3.0f}% | Temps : {mean_time:>6.2f}s (+-{std_time:>5.2f}s)")
        
    print("\nBenchmark de comparaison termine.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparaison Benchmark for L.I.A.R")
    parser.add_argument("--seeds", type=int, default=3, help="Nombre de seeds a evaluer par dimension.")
    parser.add_argument("--epochs", type=int, default=1000, help="Nombre d'epochs d'entraînement par seed (défaut: 1000).")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate Adam (défaut: 0.005).")
    parser.add_argument("--batch", type=int, default=256, help="Batch size (défaut: 256).")
    parser.add_argument("--steps", type=int, default=5, help="Nombre de pas d'attracteur pour LIAR (défaut: 5).")
    parser.add_argument("--latent-dim", type=int, default=None, help="Dimension latente (bas-rang). Par défaut: min(16, n_bits//2).")
    parser.add_argument("--struct", type=float, default=0.001, help="Coefficient de pénalité structurelle (et wave si activée) (défaut: 0.001).")
    parser.add_argument("--disable-wave", action="store_true", help="Désactive l'ajout de wave penalty dans la loss.")
    args = parser.parse_args()
    
    run_comparison_benchmark(
        args.seeds,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        steps=args.steps,
        latent_dim=args.latent_dim,
        struct_coeff=args.struct,
        disable_wave=args.disable_wave,
    )
