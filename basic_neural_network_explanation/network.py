import numpy as np
from datetime import datetime

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_neural_network_log():
    # Create log content
    log_content = []
    
    # Header
    log_content.append("=" * 80)
    log_content.append("🧠 NEURAL NETWORK FORWARD PASS CALCULATION LOG")
    log_content.append("=" * 80)
    log_content.append(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append(f"🏗️  Architecture: 4 Input → 3 Hidden → 2 Output")
    log_content.append(f"⚡ Activation Function: Sigmoid")
    log_content.append("")
    
    # Input data
    x = np.array([2, 3, -1, 2])
    W1 = np.array([
        [2, 3, 0, 1],
        [4, 3, 2, 0],
        [1, 1, 1, 2]
    ])
    W2 = np.array([
        [1, 7, 1],
        [7, 1, 4]
    ])
    b = np.array([-6, -2])
    
    # Input section
    log_content.append("📊 INPUT DATA")
    log_content.append("-" * 40)
    log_content.append(f"🔢 Input Vector x: {x.tolist()}")
    log_content.append("")
    log_content.append("🔗 Weight Matrix W1 (Input → Hidden):")
    for i, row in enumerate(W1):
        log_content.append(f"   Row {i+1}: {row.tolist()}")
    log_content.append("")
    log_content.append("🔗 Weight Matrix W2 (Hidden → Output):")
    for i, row in enumerate(W2):
        log_content.append(f"   Row {i+1}: {row.tolist()}")
    log_content.append("")
    log_content.append(f"⚖️  Bias Vector b: {b.tolist()}")
    log_content.append("")
    
    # Forward pass calculations
    log_content.append("🚀 FORWARD PASS COMPUTATION")
    log_content.append("-" * 40)
    
    # Layer 1: Input to Hidden
    log_content.append("📍 STEP 1: Input → Hidden Layer")
    z1 = W1 @ x
    log_content.append(f"   🧮 Pre-activation z1 = W1 × x")
    for i in range(3):
        calculation_parts = []
        for j in range(4):
            calculation_parts.append(f"{W1[i,j]}×{x[j]}")
        calculation = " + ".join(calculation_parts)
        log_content.append(f"   z1[{i+1}] = {calculation} = {z1[i]}")
    log_content.append("")
    
    a1 = sigmoid(z1)
    log_content.append("   ⚡ Applying Sigmoid Activation:")
    for i in range(3):
        log_content.append(f"   a1[{i+1}] = σ({z1[i]}) = {a1[i]:.6f}")
    log_content.append("")
    
    # Layer 2: Hidden to Output
    log_content.append("📍 STEP 2: Hidden → Output Layer")
    z2_no_bias = W2 @ a1
    z2 = z2_no_bias + b
    log_content.append(f"   🧮 Pre-activation z2 = W2 × a1 + b")
    for i in range(2):
        calculation_parts = []
        for j in range(3):
            calculation_parts.append(f"{W2[i,j]}×{a1[j]:.6f}")
        calculation = " + ".join(calculation_parts)
        log_content.append(f"   z2[{i+1}] = {calculation} + {b[i]} = {z2[i]:.6f}")
    log_content.append("")
    
    a2 = sigmoid(z2)
    log_content.append("   ⚡ Applying Sigmoid Activation:")
    for i in range(2):
        log_content.append(f"   a2[{i+1}] = σ({z2[i]:.6f}) = {a2[i]:.6f}")
    log_content.append("")
    
    # Final results
    log_content.append("🎯 FINAL RESULTS")
    log_content.append("-" * 40)
    log_content.append(f"🔹 Hidden Layer Output: {[f'{val:.6f}' for val in a1]}")
    log_content.append(f"🔹 Network Final Output: {[f'{val:.6f}' for val in a2]}")
    log_content.append("")
    
    # Mathematical formulas
    log_content.append("📐 MATHEMATICAL FORMULAS REFERENCE")
    log_content.append("-" * 40)
    log_content.append("🔸 Linear Transformation: z = W × a + b")
    log_content.append("🔸 Sigmoid Activation: σ(x) = 1 / (1 + e^(-x))")
    log_content.append("🔸 Forward Propagation:")
    log_content.append("   • Hidden Layer: a1 = σ(W1 × x)")
    log_content.append("   • Output Layer: a2 = σ(W2 × a1 + b)")
    log_content.append("")
    
    # Network summary
    log_content.append("📋 NETWORK SUMMARY")
    log_content.append("-" * 40)
    log_content.append(f"📌 Total Parameters: {W1.size + W2.size + b.size}")
    log_content.append(f"📌 W1 Parameters: {W1.size} (3×4)")
    log_content.append(f"📌 W2 Parameters: {W2.size} (2×3)")
    log_content.append(f"📌 Bias Parameters: {b.size}")
    log_content.append(f"📌 Network Depth: 3 layers")
    log_content.append("")
    
    # Footer
    log_content.append("=" * 80)
    log_content.append("✅ NEURAL NETWORK COMPUTATION COMPLETED SUCCESSFULLY")
    log_content.append("=" * 80)
    
    return "\n".join(log_content)

# Generate and save the log
if __name__ == "__main__":
    log_text = create_neural_network_log()
    # Write to file (no timestamped version)
    with open('neural_network_log.txt', 'w', encoding='utf-8') as f:
        f.write(log_text)