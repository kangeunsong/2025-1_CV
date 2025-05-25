# 분석 함수들

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from scipy import stats
import os

# Dead ReLU 뉴런 분석
def analyze_dead_relu(model, data_loader, device):
    dead_neurons = {}  # dead neuron 수를 저장할 딕셔너리
    total_activations = {}  # 총 activation 수를 저장할 딕셔너리
    
    # Hook을 사용해 각 레이어의 출력값 저장
    activations = {}
    
    def get_activation(name):
        # Hook 함수: 레이어의 출력을 activations 딕셔너리에 저장
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 각 레이어에 hook 등록
    hooks = []
    for i, layer in enumerate(model.layers):
        hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
        hooks.append(hook)
    
    # 모델을 평가 모드로 전환 후 데이터를 통과시키며 dead neuron 계산
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            _ = model(inputs)
            
            for name, activation in activations.items():
                # 레이어별 초기화
                if name not in dead_neurons:
                    dead_neurons[name] = torch.zeros(activation.size(1)).to(device)
                    total_activations[name] = 0
                
                # ReLU 출력이 0인 뉴런 카운트
                if model.activation_name == 'relu':
                    dead_neurons[name] += (activation == 0).float().sum(dim=0)
                elif model.activation_name == 'leaky_relu':
                    # LeakyReLU: 작은 값도 dead로 간주
                    dead_neurons[name] += (activation.abs() < 1e-8).float().sum(dim=0)
                
                # 총 activation 개수 추가
                total_activations[name] += activation.size(0)
    
    # hook 제거
    for hook in hooks:
        hook.remove()
    
    # dead neuron 비율 계산 (% 단위)
    dead_ratios = {}
    for name in dead_neurons:
        dead_ratios[name] = ((dead_neurons[name] / total_activations[name]).cpu().numpy() * 100)
    
    return dead_ratios

# Dead neuron 히트맵 시각화
def visualize_dead_neurons(dead_ratios, save_path=None):
    # 각 레이어별 dead neuron 비율을 리스트로 정리
    data = []
    layer_names = []
    max_neurons = 0
    
    for layer_name, ratios in dead_ratios.items():
        data.append(ratios)
        layer_names.append(layer_name.replace('_', ' ').title())
        max_neurons = max(max_neurons, len(ratios))
    
    # 레이어마다 뉴런 수가 다를 수 있어 NaN으로 패딩
    padded_data = []
    for layer_data in data:
        if len(layer_data) < max_neurons:
            padded = np.pad(layer_data, (0, max_neurons - len(layer_data)), 
                          constant_values=np.nan)
        else:
            padded = layer_data
        padded_data.append(padded)
    
    data = np.array(padded_data)
    
    # 히트맵 그리기
    plt.figure(figsize=(14, 6))
    
    # NaN은 마스크 처리
    mask = np.isnan(data)
    
    # seaborn 히트맵
    ax = sns.heatmap(data, 
                     mask=mask,
                     cmap='Reds', 
                     vmin=0, 
                     vmax=100,
                     cbar_kws={'label': 'Dead Neuron Ratio (%)'},
                     yticklabels=layer_names,
                     xticklabels=False)
    
    # 각 레이어의 평균 dead neuron 비율 표시
    for i, layer_name in enumerate(layer_names):
        layer_data = data[i][~np.isnan(data[i])]
        mean_ratio = np.mean(layer_data)
        ax.text(-2, i + 0.5, f'{mean_ratio:.1f}%', 
               ha='right', va='center', fontweight='bold')
    
    plt.title('Dead ReLU Neurons Heatmap\n(Darker = Higher percentage of dead neurons)', 
             fontsize=14)
    plt.xlabel('Neuron Index', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    
    # 범례 추가
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Active (0%)'),
        Rectangle((0, 0), 1, 1, facecolor='#ffcccc', edgecolor='black', label='Partially Dead (~50%)'),
        Rectangle((0, 0), 1, 1, facecolor='#ff0000', edgecolor='black', label='Dead (100%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # 저장 옵션
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Gradient 흐름 분석
def analyze_gradient_flow(model, data_loader, device):
    model.train()  # 학습 모드 설정
    
    # 한 배치만 사용 (계산량 최소화)
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Forward pass
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    
    # Backward pass (gradient 계산)
    model.zero_grad()
    loss.backward()
    
    # gradient 통계 수집
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().cpu().numpy()
            gradient_stats[name] = {
                'mean': np.mean(np.abs(grad)),
                'std': np.std(grad),
                'max': np.max(np.abs(grad)),
                'min': np.min(np.abs(grad)),
                'zeros_ratio': np.sum(grad == 0) / grad.size * 100
            }
    
    return gradient_stats

# Gradient flow 시각화
def visualize_gradient_flow(gradient_stats, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 레이어별 이름과 gradient 통계 추출
    layer_names = []
    means = []
    stds = []
    zeros_ratios = []
    
    for name, stats in gradient_stats.items():
        layer_names.append(name.replace('.weight', '').replace('.bias', ' (b)'))
        means.append(stats['mean'])
        stds.append(stats['std'])
        zeros_ratios.append(stats['zeros_ratio'])
    
    x = np.arange(len(layer_names))
    
    # 평균 gradient magnitude + 표준편차 시각화
    ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax1.set_ylabel('Average Gradient Magnitude', fontsize=12)
    ax1.set_title('Gradient Flow Through Network', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    ax1.set_yscale('log')  # gradient 크기 분포를 더 잘 보여주기 위해 로그 스케일 사용
    ax1.grid(True, alpha=0.3)
    
    # gradient가 매우 작은 레이어 강조
    for i, mean in enumerate(means):
        if mean < 1e-5:
            ax1.bar(i, mean, color='red', alpha=0.8)
    
    # Zero gradient 비율 시각화
    ax2.bar(x, zeros_ratios, alpha=0.7, color='orange')
    ax2.set_ylabel('Zero Gradients (%)', fontsize=12)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_title('Percentage of Zero Gradients by Layer', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 파일 저장 옵션
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 활성화 값 분포 시각화
def visualize_activation_distribution(model, data_loader, device, layer_idx=0, save_path=None):
    activations = []  # 활성화 값 저장
    
    def hook(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    # Hook 등록 (지정한 레이어만)
    if layer_idx < len(model.layers):
        handle = model.layers[layer_idx].register_forward_hook(hook)
    else:
        print(f"Warning: layer_idx {layer_idx} out of range")
        return None
    
    # 모델 평가 모드로 전환 후 10 배치만 분석
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            if len(activations) >= 10:
                break
    
    handle.remove()
    
    # 활성화 값 전처리
    act_values = np.concatenate(activations).flatten()
    
    # 통계 계산
    activation_stats = {
        'mean': np.mean(act_values),
        'std': np.std(act_values),
        'min': np.min(act_values),
        'max': np.max(act_values),
        'zeros_ratio': np.sum(act_values == 0) / len(act_values) * 100,
        'negative_ratio': np.sum(act_values < 0) / len(act_values) * 100
    }
    
    # 히스토그램 및 Box plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(act_values, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero')
    ax1.axvline(x=activation_stats['mean'], color='green', linestyle='--', 
               alpha=0.5, label=f"Mean: {activation_stats['mean']:.3f}")
    ax1.set_xlabel('Activation Value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Activation Distribution - Layer {layer_idx + 1}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.boxplot(act_values, vert=True)
    ax2.set_ylabel('Activation Value', fontsize=12)
    ax2.set_title(f'Activation Range - Layer {layer_idx + 1}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    stats_text = f"Mean: {activation_stats['mean']:.3f}\n"
    stats_text += f"Std: {activation_stats['std']:.3f}\n"
    stats_text += f"Min: {activation_stats['min']:.3f}\n"
    stats_text += f"Max: {activation_stats['max']:.3f}\n"
    stats_text += f"Zeros: {activation_stats['zeros_ratio']:.1f}%"
    
    ax2.text(1.5, activation_stats['mean'], stats_text, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.suptitle(f'{model.activation_name.upper()} Activation Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return activation_stats

# 가중치 분포 분석
def analyze_weight_distribution(model, save_path=None):
    weights = model.get_layer_weights()
    
    fig, axes = plt.subplots(1, len(weights), figsize=(5 * len(weights), 5))
    if len(weights) == 1:
        axes = [axes]
    
    for i, (w, ax) in enumerate(zip(weights, axes)):
        w_flat = w.flatten()
        
        ax.hist(w_flat, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(w_flat), color='green', linestyle='--', alpha=0.5)
        
        mu, std = stats.norm.fit(w_flat)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, label=f'Normal fit\nμ={mu:.3f}, σ={std:.3f}')
        
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Layer {i+1} Weights\nShape: {w.shape}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weight Distribution Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 모델들의 수렴 속도 비교
def compare_models_convergence(results_dict, threshold=90, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    convergence_data = []
    
    for name, results in results_dict.items():
        accs = results.get('test_accuracies', results.get('accuracies', []))
        
        conv_epoch = next((i+1 for i, acc in enumerate(accs) if acc >= threshold), None)
        
        if conv_epoch:
            convergence_data.append({
                'Method': name,
                'Convergence Epoch': conv_epoch,
                'Final Accuracy': accs[-1]
            })
            
            epochs = range(1, conv_epoch + 6)
            plot_epochs = min(len(epochs), len(accs))
            ax.plot(epochs[:plot_epochs], accs[:plot_epochs], 
                   marker='o', markersize=4, linewidth=2, label=name)
            
            ax.scatter([conv_epoch], [accs[conv_epoch-1]], s=100, 
                      marker='*', edgecolors='black', linewidth=2)
    
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, 
              label=f'{threshold}% Threshold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    if convergence_data:
        df = pd.DataFrame(convergence_data)
        print("\nConvergence Analysis:")
        print(df.to_string(index=False))
    
    return convergence_data

# 전체 실험 보고서 생성
def generate_experiment_report(all_results, results_dir):
    report_path = os.path.join(results_dir, "analysis", "experiment_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("컴퓨터 비전 과제 3: MLP 실험 분석 보고서\n")
        f.write("="*80 + "\n\n")
        
        # 실험 A 분석
        if 'experiment_A' in all_results:
            f.write("## 실험 A: 손실 함수 비교 분석\n")
            f.write("-"*40 + "\n")
            
            ce_results = all_results['experiment_A']['CrossEntropy']
            mse_results = all_results['experiment_A']['MSE_with_Softmax']
            
            f.write(f"CrossEntropy 최종 정확도: {ce_results['final_accuracy']:.2f}%\n")
            f.write(f"MSE (with Softmax) 최종 정확도: {mse_results['final_accuracy']:.2f}%\n")
            f.write(f"성능 차이: {ce_results['final_accuracy'] - mse_results['final_accuracy']:.2f}%\n\n")
            
            f.write("### MSE가 CrossEntropy보다 학습이 느린 이유:\n")
            f.write("1. Gradient 특성 차이:\n")
            f.write("   - MSE: ∂L/∂z = 2(p-y) * p(1-p) (softmax 미분항 포함)\n")
            f.write("   - CE: ∂L/∂z = p-y (단순한 형태)\n")
            f.write("2. Softmax 미분항 p(1-p)는 p가 0 또는 1에 가까울 때 0에 가까워짐\n")
            f.write("3. 이로 인해 MSE는 gradient vanishing 문제 발생\n\n")
        
        # 실험 B 분석
        if 'experiment_B' in all_results:
            f.write("\n## 실험 B: 활성화 함수 비교 분석\n")
            f.write("-"*40 + "\n")
            
            for activation, results in all_results['experiment_B'].items():
                f.write(f"\n{activation.upper()}:\n")
                f.write(f"  - 최종 정확도: {results['final_accuracy']:.2f}%\n")
                f.write(f"  - Dead neurons: {results['dead_ratio']:.2f}%\n")
                f.write(f"  - 수렴 에폭: {results['convergence_epoch']}\n")
            
            f.write("\n### Dead ReLU 현상 분석:\n")
            f.write("1. 작은 초기화 값(std=0.01) 사용으로 Dead ReLU 유도\n")
            f.write("2. ReLU: 음수 입력에 대해 gradient가 완전히 0\n")
            f.write("3. LeakyReLU: 음수 구간에서도 작은 gradient(0.01) 유지\n")
            f.write("4. Sigmoid: Gradient vanishing 문제 있지만 dead neuron은 없음\n\n")
        
        # 실험 C 분석
        if 'experiment_C' in all_results:
            f.write("\n## 실험 C: 최적화 알고리즘 비교 분석\n")
            f.write("-"*40 + "\n")
            
            f.write("\n### 스케줄러 효과 분석:\n")
            
            best_acc = 0
            best_config = ""
            
            for scheduler_use, results in all_results['experiment_C'].items():
                for name, metrics in results.items():
                    if not metrics.get('diverged', False) and metrics['final_accuracy'] > best_acc:
                        best_acc = metrics['final_accuracy']
                        best_config = f"{name} ({'with' if 'with' in scheduler_use else 'without'} scheduler)"
            
            f.write(f"최고 성능: {best_config} - {best_acc:.2f}%\n\n")
            
            f.write("### Optimizer별 특성:\n")
            f.write("1. SGD: 학습률에 매우 민감, momentum 없이는 수렴 느림\n")
            f.write("2. SGD+Momentum: 안정적인 학습, 적절한 학습률 필요\n")
            f.write("3. Adam: Adaptive learning rate로 다양한 학습률에서 안정적\n")
            f.write("   - 높은 학습률(0.1)에서는 스케줄러 필수\n\n")
        
        f.write("\n## 종합 결론\n")
        f.write("-"*40 + "\n")
        f.write("1. 손실 함수: 다중 클래스 분류에는 CrossEntropy가 MSE보다 효과적\n")
        f.write("2. 활성화 함수: LeakyReLU가 ReLU의 dead neuron 문제를 해결하면서 성능 유지\n")
        f.write("3. 최적화: Adam + 적절한 학습률(0.01) + 스케줄러 조합이 최고 성능\n")
        f.write("4. 일반화: 모든 실험에서 과적합은 크게 관찰되지 않음\n")
    
    print(f"\n분석 보고서가 {report_path}에 저장되었습니다.")

# 모든 실험 결과 종합 시각화
def plot_all_experiments_summary(all_results, results_dir):
    fig = plt.figure(figsize=(20, 12))
    
    # 실험 A 요약
    ax1 = plt.subplot(2, 3, 1)
    if 'experiment_A' in all_results:
        methods = list(all_results['experiment_A'].keys())
        accuracies = [results['final_accuracy'] for results in all_results['experiment_A'].values()]
        bars = ax1.bar(methods, accuracies, color=['blue', 'red'])
        ax1.set_ylabel('Final Accuracy (%)')
        ax1.set_title('Experiment A: Loss Functions')
        ax1.set_ylim(80, 100)
        
        # 값 표시
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom')
    
    # 실험 B 요약
    ax2 = plt.subplot(2, 3, 2)
    if 'experiment_B' in all_results:
        methods = list(all_results['experiment_B'].keys())
        accuracies = [results['final_accuracy'] for results in all_results['experiment_B'].values()]
        dead_ratios = [results['dead_ratio'] for results in all_results['experiment_B'].values()]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='green')
        bars2 = ax2.bar(x + width/2, dead_ratios, width, label='Dead Neurons (%)', color='red')
        
        ax2.set_xlabel('Activation Function')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Experiment B: Activation Functions')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.upper() for m in methods])
        ax2.legend()
        
        # 값 표시
        for bar, val in zip(bars1, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, dead_ratios):
            if val > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 실험 C 요약 (최적 조합들만)
    ax3 = plt.subplot(2, 3, 3)
    if 'experiment_C' in all_results:
        # 각 optimizer의 최고 성능만 추출
        best_results = {}
        
        for scheduler_use, results in all_results['experiment_C'].items():
            for name, metrics in results.items():
                if not metrics.get('diverged', False):
                    opt_name = name.split('_')[0]
                    if 'Momentum' in name:
                        opt_name = 'SGD_Momentum'
                    
                    key = f"{opt_name}_{'w/' if 'with' in scheduler_use else 'w/o'}_scheduler"
                    
                    if key not in best_results or metrics['final_accuracy'] > best_results[key]:
                        best_results[key] = metrics['final_accuracy']
        
        methods = list(best_results.keys())
        accuracies = list(best_results.values())
        
        bars = ax3.bar(methods, accuracies)
        ax3.set_xlabel('Optimizer Configuration')
        ax3.set_ylabel('Best Accuracy (%)')
        ax3.set_title('Experiment C: Optimizers (Best Configurations)')
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        ax3.set_ylim(80, 100)
        
        # 값 표시
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom')
    
    # 전체 최고 성능 비교
    ax4 = plt.subplot(2, 1, 2)
    
    all_best = []
    
    # 실험 A 최고
    if 'experiment_A' in all_results:
        best_a = max(all_results['experiment_A'].items(), 
                    key=lambda x: x[1]['final_accuracy'])
        all_best.append({
            'Experiment': 'A: Loss Function',
            'Configuration': best_a[0],
            'Accuracy': best_a[1]['final_accuracy']
        })
    
    # 실험 B 최고
    if 'experiment_B' in all_results:
        best_b = max(all_results['experiment_B'].items(), 
                    key=lambda x: x[1]['final_accuracy'])
        all_best.append({
            'Experiment': 'B: Activation',
            'Configuration': best_b[0],
            'Accuracy': best_b[1]['final_accuracy']
        })
    
    # 실험 C 최고
    if 'experiment_C' in all_results:
        best_c_acc = 0
        best_c_config = ""
        
        for scheduler_use, results in all_results['experiment_C'].items():
            for name, metrics in results.items():
                if not metrics.get('diverged', False) and metrics['final_accuracy'] > best_c_acc:
                    best_c_acc = metrics['final_accuracy']
                    best_c_config = f"{name} ({'with' if 'with' in scheduler_use else 'without'} scheduler)"
        
        all_best.append({
            'Experiment': 'C: Optimizer',
            'Configuration': best_c_config,
            'Accuracy': best_c_acc
        })
    
    if all_best:
        df_best = pd.DataFrame(all_best)
        
        # 바 차트
        bars = ax4.bar(df_best['Experiment'], df_best['Accuracy'])
        ax4.set_ylabel('Best Accuracy (%)')
        ax4.set_title('Best Performance Across All Experiments', fontsize=16)
        ax4.set_ylim(90, 100)
        
        # 값과 설정 표시
        for i, (bar, row) in enumerate(zip(bars, df_best.iterrows())):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{row[1]["Accuracy"]:.2f}%\n({row[1]["Configuration"]})',
                    ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('MLP Experiments Summary', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "all_experiments_summary.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()

# 학습 동역학 분석 (수렴 속도, 진동, 정체 구간 등)
def analyze_learning_dynamics(train_losses, test_accuracies, window_size=5):
    dynamics = {}
    
    # 수렴 속도: 90% 정확도 달성 에폭
    dynamics['convergence_epoch'] = next(
        (i+1 for i, acc in enumerate(test_accuracies) if acc > 90), 
        len(test_accuracies)
    )
    
    # 손실 감소율 (감소 구간만 평균)
    loss_gradients = np.gradient(train_losses)
    dynamics['avg_loss_decrease_rate'] = np.mean(loss_gradients[loss_gradients < 0])
    
    # 정확도 증가율 (증가 구간만 평균)
    acc_gradients = np.gradient(test_accuracies)
    dynamics['avg_accuracy_increase_rate'] = np.mean(acc_gradients[acc_gradients > 0])
    
    # 손실 진동 정도 (이동평균과의 표준편차)
    if len(train_losses) > window_size:
        loss_ma = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        dynamics['loss_oscillation'] = np.std(np.array(train_losses[window_size-1:]) - loss_ma)
    else:
        dynamics['loss_oscillation'] = np.std(train_losses)
    
    # 정체 구간 찾기 (정확도 변화량이 작은 구간)
    stagnant_epochs = []
    for i in range(len(test_accuracies) - window_size):
        window = test_accuracies[i:i+window_size]
        if np.std(window) < 0.5:  # 0.5% 미만의 변화
            stagnant_epochs.append(i + window_size//2)
    
    dynamics['stagnant_epochs'] = stagnant_epochs
    dynamics['num_stagnant_periods'] = len(stagnant_epochs)
    
    # 최종 성능
    dynamics['final_loss'] = train_losses[-1]
    dynamics['final_accuracy'] = test_accuracies[-1]
    dynamics['best_accuracy'] = max(test_accuracies)
    dynamics['best_accuracy_epoch'] = test_accuracies.index(max(test_accuracies)) + 1
    
    return dynamics