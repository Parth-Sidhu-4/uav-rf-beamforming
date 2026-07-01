import torch
import time
import numpy as np
import sys

class SirenCovarianceNet(torch.nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=160, w0=30.0):
        super().__init__()
        self.net = []
        self.net.append(torch.nn.Linear(in_features, hidden_features))
        for _ in range(hidden_layers):
            self.net.append(torch.nn.Linear(hidden_features, hidden_features))
        self.net.append(torch.nn.Linear(hidden_features, out_features))
        self.net = torch.nn.ModuleList(self.net)
        self.w0 = w0

    def forward(self, x):
        out = self.net[0](x)
        out = torch.sin(self.w0 * out)
        for i in range(1, len(self.net) - 1):
            out = self.net[i](out)
            out = torch.sin(self.w0 * out)
        out = self.net[-1](out)
        return out

def benchmark():
    print("--- UAV Beamforming: End-to-End Latency Benchmark ---")
    
    device = torch.device('cpu')
    model = SirenCovarianceNet(in_features=2, hidden_features=256, hidden_layers=3, out_features=160).to(device)
    model.eval()
    
    batch_sizes = [1, 10, 100]
    
    print("\n[Warming up CPU...]")
    for _ in range(100):
        _ = model(torch.rand(1, 2))
        
    for b in batch_sizes:
        print(f"\n--- Batch Size: {b} ---")
        x = torch.rand(b, 2).to(device)
        
        times = []
        with torch.no_grad():
            for _ in range(1000):
                start = time.perf_counter()
                
                out = model(x)
                V = out.view(b, 32, 5)
                R = torch.bmm(V, V.transpose(1, 2)) 
                eye = torch.eye(32).unsqueeze(0).expand(b, -1, -1)
                R = R + 1e-3 * eye
                
                v_sig = torch.ones(b, 32, 1, dtype=torch.float32)
                try:
                    w = torch.linalg.solve(R, v_sig)
                except Exception as e:
                    pass
                    
                end = time.perf_counter()
                times.append((end - start) * 1000)
                
        times = np.array(times)
        print(f"Mean Latency: {np.mean(times):.3f} ms")
        print(f"Min Latency:  {np.min(times):.3f} ms")
        print(f"99th % Latency: {np.percentile(times, 99):.3f} ms")

if __name__ == "__main__":
    benchmark()
