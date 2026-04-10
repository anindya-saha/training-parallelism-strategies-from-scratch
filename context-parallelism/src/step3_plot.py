"""Step 3: Generate comparison plots"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('outputs/cp_results.json') as f:
    results = json.load(f)

seq_lens = [int(s) for s in results.keys()]
no_cp_mem = [results[str(s)]['no_cp']['mem_mb'] for s in seq_lens]
cp_mem = [results[str(s)]['with_cp']['mem_mb'] for s in seq_lens]
reductions = [results[str(s)]['reduction_pct'] for s in seq_lens]
no_cp_time = [results[str(s)]['no_cp']['time_ms'] for s in seq_lens]
cp_time = [results[str(s)]['with_cp']['time_ms'] for s in seq_lens]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Context Parallelism: Performance Analysis', fontsize=16, fontweight='bold', y=0.98)

colors = {'no_cp': '#E74C3C', 'cp': '#2ECC71', 'blue': '#3498DB'}

# ============================================================
# Plot 1: Memory Comparison Bar Chart
# ============================================================
ax1 = axes[0, 0]
x = np.arange(len(seq_lens))
width = 0.35

bars1 = ax1.bar(x - width/2, no_cp_mem, width, label='No CP', color=colors['no_cp'], edgecolor='black', linewidth=1)
bars2 = ax1.bar(x + width/2, cp_mem, width, label='With CP', color=colors['cp'], edgecolor='black', linewidth=1)

ax1.set_xlabel('Sequence Length', fontsize=11)
ax1.set_ylabel('Peak Memory (MB)', fontsize=11)
ax1.set_title('Memory Usage: No CP vs With CP', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(seq_lens)
ax1.legend(loc='upper left')
ax1.set_ylim(0, max(no_cp_mem) * 1.25)

for bar in bars1:
    ax1.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
for bar in bars2:
    ax1.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

# ============================================================
# Plot 2: Memory Reduction Percentage
# ============================================================
ax2 = axes[0, 1]
bars = ax2.bar(seq_lens, reductions, color=colors['blue'], edgecolor='black', linewidth=1, width=300)
ax2.set_xlabel('Sequence Length', fontsize=11)
ax2.set_ylabel('Memory Reduction (%)', fontsize=11)
ax2.set_title('Memory Savings with Context Parallelism', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(reductions) * 1.3)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% target')

for bar, val in zip(bars, reductions):
    ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

# ============================================================
# Plot 3: Time Comparison
# ============================================================
ax3 = axes[1, 0]
bars1 = ax3.bar(x - width/2, no_cp_time, width, label='No CP', color=colors['no_cp'], edgecolor='black', linewidth=1)
bars2 = ax3.bar(x + width/2, cp_time, width, label='With CP', color=colors['cp'], edgecolor='black', linewidth=1)

ax3.set_xlabel('Sequence Length', fontsize=11)
ax3.set_ylabel('Time (ms)', fontsize=11)
ax3.set_title('Computation Time per Iteration', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(seq_lens)
ax3.legend(loc='upper left')

for bar in bars1:
    ax3.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
for bar in bars2:
    ax3.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)

# ============================================================
# Plot 4: Theoretical Scaling
# ============================================================
ax4 = axes[1, 1]

# Theoretical memory: attention = B * H * S * S * 4 bytes
theo_seq = [512, 1024, 2048, 4096, 8192, 16384, 32768]
B, H = 4, 16

for cp_deg, color, style in [(1, colors['no_cp'], '-'), (2, colors['cp'], '--'), (4, '#9B59B6', ':')]:
    mem = [B * H * (s/cp_deg) * s * 4 / 1024**2 for s in theo_seq]  # MB
    label = f'CP={cp_deg}' if cp_deg > 1 else 'No CP'
    ax4.plot(theo_seq, mem, color=color, linestyle=style, linewidth=2.5, marker='o', markersize=6, label=label)

ax4.axhline(y=40000, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax4.text(theo_seq[-1]*0.6, 42000, '40GB GPU Limit', fontsize=10, color='red')

ax4.set_xlabel('Sequence Length', fontsize=11)
ax4.set_ylabel('Attention Memory (MB)', fontsize=11)
ax4.set_title('Memory Scaling (Theoretical)', fontsize=12, fontweight='bold')
ax4.set_xscale('log', base=2)
ax4.set_yscale('log')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(theo_seq)
ax4.set_xticklabels([f'{s//1024}K' if s >= 1024 else str(s) for s in theo_seq])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('images/cp_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print('✓ Saved: images/cp_comparison.png')
plt.show()
