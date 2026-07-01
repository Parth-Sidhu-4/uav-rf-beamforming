import numpy as np
import trimesh
import sys
import glob
import ast



# Load all genomes from eval_logs
log_files = glob.glob("eval_log_*.txt")
genomes = []

for f in log_files:
    with open(f, 'r') as file:
        for line in file:
            parts = line.strip().split("],")
            if len(parts) == 2:
                genome_str = parts[0] + "]"
                fitness_str = parts[1]
                try:
                    genome = ast.literal_eval(genome_str)
                    fitness = float(fitness_str)
                    genomes.append((genome, fitness))
                except Exception as e:
                    pass

print(f"Total evaluated genomes loaded: {len(genomes)}")

# Deduplicate
unique_genomes = {}
for g, f in genomes:
    g_tuple = tuple(g)
    if g_tuple not in unique_genomes or f < unique_genomes[g_tuple]:
        unique_genomes[g_tuple] = f

# Sort by fitness
sorted_genomes = sorted(unique_genomes.items(), key=lambda x: x[1])
top_10 = sorted_genomes[:10]

print("\nTop 10 Surrogate Genomes:")
for i, (g, f) in enumerate(top_10):
    print(f"{i+1}. Genome: {g} | Fitness: {-f:.3f}")

print("\nSaving Top 10 to top10_genomes.npy")
np.save("top10_genomes.npy", [g for g, f in top_10])
