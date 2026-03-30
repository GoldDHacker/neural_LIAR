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

def generate_parity_data(batch_size, n_bits, device):
    X = torch.randint(0, 2, (batch_size, n_bits), device=device).float() * 2 - 1
    Y = torch.prod(X, dim=1, keepdim=True)
    return X, Y

def train_liar_seed(n_bits, seed, epochs=500, batch_size=256, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = LIARNeuron(
        in_features=n_bits, 
        out_features=1, 
        max_steps=5, 
        latent_dim=min(16, n_bits//2), 
        max_order=3
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        X, Y = generate_parity_data(batch_size, n_bits, device)
        
        Z_out = model(X)
        out = Z_out[:, 0:1]
        
        loss = criterion(out, Y) + 0.001 * model.get_structural_penalty()
        if hasattr(model, 'get_wave_penalty'):
            loss += 0.001 * model.get_wave_penalty()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        X_test, Y_test = generate_parity_data(2000, n_bits, device)
        Z_out = model(X_test)
        out_test = Z_out[:, 0:1]
        
        preds = torch.sign(out_test)
        correct = (preds == Y_test).float().sum().item()
        accuracy = (correct / 2000.0) * 100
        
    return accuracy

def run_robustness_benchmark(num_seeds):
    print(f"\n{'='*80}")
    print(f" BENCHMARK DE ROBUSTESSE STATISTIQUE : L.I.A.R v2 sur Parite Globale")
    print(f" Nombres de Seeds par Dimension : {num_seeds}")
    print(f"{'='*80}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Appareil d'execution : {device}")
    
    N_BITS_LIST = [4, 8, 16, 32]
    
    for n_bits in N_BITS_LIST:
        print(f"\n[ Test N={n_bits} Bits | Entrainement sur {num_seeds} seeds ]")
        print("-" * 50)
        
        accuracies = []
        for seed in range(num_seeds):
            start_t = time.time()
            acc = train_liar_seed(n_bits, seed, epochs=500, batch_size=256, device=device)
            elapsed = time.time() - start_t
            
            marker = "[OK]" if acc > 90.0 else "[FAIL]"
            print(f" Seed {seed+1:02d} | Accuracy : {acc:>6.1f}% {marker} | Temps : {elapsed:.1f}s")
            accuracies.append(acc)
            
        acc_array = np.array(accuracies)
        mean_acc = acc_array.mean()
        std_acc = acc_array.std()
        success_rate = (acc_array > 90.0).mean() * 100
        
        print("-" * 50)
        print(f" Resume N={n_bits} | Moyenne : {mean_acc:.1f}% (+-{std_acc:.1f}%) | Taux de Resolution : {success_rate:.0f}%")
    
    print("\nBenchmark termine.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Scalability Robustness Benchmark for L.I.A.R v2")
    parser.add_argument("--seeds", type=int, default=5, help="Nombre de seeds a evaluer par dimension.")
    args = parser.parse_args()
    
    run_robustness_benchmark(args.seeds)
