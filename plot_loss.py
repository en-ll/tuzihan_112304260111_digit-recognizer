import json
import matplotlib.pyplot as plt

# 读取实验结果
with open('experiment_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 绘制Loss曲线
plt.figure(figsize=(12, 8))

colors = ['b', 'g', 'r', 'm']
labels = ['Exp1 (SGD)', 'Exp2 (Adam)', 'Exp3 (Adam+ES)', 'Exp4 (Adam+DA+ES)']

for i, result in enumerate(results):
    epochs_range = range(1, len(result['train_losses']) + 1)
    plt.plot(epochs_range, result['train_losses'], f'{colors[i]}-', 
             label=labels[i], linewidth=2, marker='o', markersize=4)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss Curves Comparison - 4组对比实验', fontsize=14)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图片
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
print("Loss曲线图已保存到 loss_curves.png")

# 打印结果汇总
print("\n" + "="*80)
print("实验结果汇总表")
print("="*80)
print(f"{'Exp':<6} {'Optimizer':<8} {'LR':<8} {'BS':<6} {'DA':<6} {'ES':<6} {'Train Acc':<12} {'Val Acc':<12} {'Best Loss':<12} {'Epoch':<6}")
print("-" * 100)
for i, r in enumerate(results, 1):
    print(f"Exp{i:<4} {r['optimizer']:<8} {r['lr']:<8} {r['batch_size']:<6} {'Y' if r['use_augmentation'] else 'N':<6} {'Y' if r['use_early_stopping'] else 'N':<6} {r['train_acc']:.2f}%{'':<6} {r['val_acc']:.2f}%{'':<6} {r['best_val_loss']:.4f}{'':<6} {r['final_epoch']}")
print("="*80)
