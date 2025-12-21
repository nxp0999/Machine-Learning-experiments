import torch
import time

print("="*70)
print("CS 6375 Project 3 - Deep Learning for MNIST and CIFAR-10")
print("="*70)

# Check GPU
if torch.cuda.is_available():
    print(f"\n Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("\n Using CPU (will be slower)")

print("\nStarting experiments...\n")

# Track total time
start_time = time.time()

# Run MNIST experiments
print("\n" + "="*70)
print("Running MNIST Experiments...")
print("="*70)
try:
    import mnist_cnn
    print("✓ MNIST experiments completed")
except Exception as e:
    print(f"✗ Error in MNIST: {e}")

# Run CIFAR-10 experiments
print("\n" + "="*70)
print("Running CIFAR-10 Experiments...")
print("="*70)
try:
    import cifar10_cnn
    print("✓ CIFAR-10 experiments completed")
except Exception as e:
    print(f"✗ Error in CIFAR-10: {e}")

# Run MLP experiments (if separate)
print("\n" + "="*70)
print("Running MLP Experiments...")
print("="*70)
try:
    import mlp
    print("✓ MLP experiments completed")
except Exception as e:
    print(f"✗ Error in MLP: {e}")

# Print final summary
elapsed_time = (time.time() - start_time) / 60
print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETE!")
print("="*70)
print(f"\nTotal Time: {elapsed_time:.1f} minutes")
