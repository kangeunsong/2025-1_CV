# 실험 함수들 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import MLP, create_model
from utils import (
    get_fashion_mnist_loaders,
    get_sklearn_data_loaders,
    train_model,
    plot_learning_curves,
    plot_comparison,
    create_summary_table,
    print_summary_table,
    plot_decision_boundary,
    save_model
)
from analysis import analyze_dead_relu, visualize_dead_neurons
import os

# 실험 A: 손실 함수 비교 (CrossEntropy vs MSE)
def experiment_A_loss_comparison(device, results_dir):
    # 실험 설정 출력
    print("\n실험 설정:")
    print("- 데이터셋: Fashion-MNIST")
    print("- 네트워크: MLP [784 -> 256 -> 128 -> 10]")
    print("- 옵티마이저: Adam (lr=0.001)")
    print("- 에폭: 30")
    
    # Fashion-MNIST 데이터셋 로드 (train/test loader, 입력 크기, 클래스 수)
    train_loader, test_loader, input_size, num_classes = get_fashion_mnist_loaders(batch_size=128)
    
    results = {}  # 결과 저장용 딕셔너리
    
    # CrossEntropy Loss 실험
    print("\n1. CrossEntropy Loss 실험 시작...")
    # 모델 초기화 및 디바이스 할당
    model_ce = MLP(input_size, [256, 128], num_classes, activation='relu').to(device)
    # 옵티마이저 설정 (Adam)
    optimizer_ce = optim.Adam(model_ce.parameters(), lr=0.001)
    # 손실 함수 설정 (CrossEntropy)
    loss_fn_ce = nn.CrossEntropyLoss()
    
    # 모델 학습 및 평가
    losses_ce, accs_ce, train_accs_ce = train_model(
        model_ce, train_loader, test_loader, 
        loss_fn_ce, optimizer_ce, 
        epochs=30, device=device,
        verbose=True, log_interval=5
    )
    
    # 결과 저장
    results['CrossEntropy'] = {
        'train_losses': losses_ce,               # 에폭별 학습 손실
        'test_accuracies': accs_ce,              # 에폭별 테스트 정확도
        'train_accuracies': train_accs_ce,       # 에폭별 학습 정확도
        'final_accuracy': accs_ce[-1],           # 마지막 에폭의 테스트 정확도
        'min_loss': min(losses_ce),              # 최소 학습 손실
        'convergence_epoch': next((i+1 for i, acc in enumerate(accs_ce) if acc > 90), 30)  # 90% 달성 에폭 (없으면 30)
    }
    
    # 학습된 모델 저장
    save_model(model_ce, os.path.join(results_dir, "models", "model_ce.pth"))
    
    # MSE Loss (Softmax 포함) 실험
    print("\n2. MSE Loss (with Softmax) 실험 시작...")
    # 모델 초기화 및 디바이스 할당
    model_mse = MLP(input_size, [256, 128], num_classes, activation='relu').to(device)
    # 옵티마이저 설정 (Adam)
    optimizer_mse = optim.Adam(model_mse.parameters(), lr=0.001)
    # 손실 함수 설정 (MSE)
    loss_fn_mse = nn.MSELoss()
    
    # 모델 학습 및 평가 (use_softmax_mse 옵션 활성화)
    losses_mse, accs_mse, train_accs_mse = train_model(
        model_mse, train_loader, test_loader,
        loss_fn_mse, optimizer_mse,
        epochs=30, device=device,
        use_softmax_mse=True,
        verbose=True, log_interval=5
    )
    
    # 결과 저장
    results['MSE_with_Softmax'] = {
        'train_losses': losses_mse,
        'test_accuracies': accs_mse,
        'train_accuracies': train_accs_mse,
        'final_accuracy': accs_mse[-1],
        'min_loss': min(losses_mse),
        'convergence_epoch': next((i+1 for i, acc in enumerate(accs_mse) if acc > 90), 30)
    }
    
    # 학습된 모델 저장
    save_model(model_mse, os.path.join(results_dir, "models", "model_mse.pth"))
    
    # 결과 시각화를 위한 subplot 2x2 구성
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 학습 손실 비교 그래프
    epochs = range(1, 31)
    ax1.plot(epochs, losses_ce, 'b-', label='CrossEntropy', linewidth=2)
    ax1.plot(epochs, losses_mse, 'r-', label='MSE (with Softmax)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 테스트 정확도 비교 그래프
    ax2.plot(epochs, accs_ce, 'b-', label='CrossEntropy', linewidth=2)
    ax2.plot(epochs, accs_mse, 'r-', label='MSE (with Softmax)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # CrossEntropy: 학습/테스트 정확도 비교
    ax3.plot(epochs, train_accs_ce, 'b--', label='Train', linewidth=2)
    ax3.plot(epochs, accs_ce, 'b-', label='Test', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('CrossEntropy: Train vs Test Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # MSE: 학습/테스트 정확도 비교
    ax4.plot(epochs, train_accs_mse, 'r--', label='Train', linewidth=2)
    ax4.plot(epochs, accs_mse, 'r-', label='Test', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('MSE: Train vs Test Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 전체 제목 설정 및 레이아웃 조정
    plt.suptitle('Experiment A: Loss Function Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "experiment_A_comparison.png"), dpi=300)
    plt.show()
    
    # 결과 요약 테이블 생성 및 출력
    summary_df = create_summary_table(results, ['final_accuracy', 'convergence_epoch', 'min_loss'])
    print_summary_table(summary_df, "Experiment A: Loss Function Comparison Summary")
    
    # 수렴 속도 비교: 90% 정확도 달성 시점 확대
    plt.figure(figsize=(10, 6))
    conv_epochs = min(results['CrossEntropy']['convergence_epoch'], 
                     results['MSE_with_Softmax']['convergence_epoch']) + 5
    plt.plot(epochs[:conv_epochs], accs_ce[:conv_epochs], 'b-', 
             label='CrossEntropy', linewidth=3)
    plt.plot(epochs[:conv_epochs], accs_mse[:conv_epochs], 'r-', 
             label='MSE (with Softmax)', linewidth=3)
    plt.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='90% Threshold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Convergence Speed Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "plots", "experiment_A_convergence.png"), dpi=300)
    plt.show()
    
    return results  # 모든 결과 반환

# 실험 A 결과
"""
============================================================
실험 A: 손실 함수 비교 (CrossEntropy vs MSE)
============================================================

실험 설정:
- 데이터셋: Fashion-MNIST
- 네트워크: MLP [784 -> 256 -> 128 -> 10]
- 옵티마이저: Adam (lr=0.001)
- 에폭: 30

1. CrossEntropy Loss 실험 시작...
Epoch [5/30], Loss: 0.2816, Train Acc: 89.45%, Test Acc: 88.34%, LR: 0.001000
Epoch [10/30], Loss: 0.2163, Train Acc: 91.92%, Test Acc: 88.39%, LR: 0.001000
Epoch [15/30], Loss: 0.1725, Train Acc: 93.42%, Test Acc: 88.37%, LR: 0.001000
Epoch [20/30], Loss: 0.1365, Train Acc: 94.85%, Test Acc: 88.38%, LR: 0.001000
Epoch [25/30], Loss: 0.1105, Train Acc: 95.84%, Test Acc: 88.48%, LR: 0.001000
Epoch [30/30], Loss: 0.0921, Train Acc: 96.48%, Test Acc: 88.68%, LR: 0.001000

2. MSE Loss (with Softmax) 실험 시작...
Epoch [5/30], Loss: 0.0157, Train Acc: 89.21%, Test Acc: 86.63%, LR: 0.001000
Epoch [10/30], Loss: 0.0129, Train Acc: 91.36%, Test Acc: 87.98%, LR: 0.001000
Epoch [15/30], Loss: 0.0110, Train Acc: 92.77%, Test Acc: 88.74%, LR: 0.001000
Epoch [20/30], Loss: 0.0095, Train Acc: 93.82%, Test Acc: 88.83%, LR: 0.001000
Epoch [25/30], Loss: 0.0081, Train Acc: 94.84%, Test Acc: 88.19%, LR: 0.001000
Epoch [30/30], Loss: 0.0073, Train Acc: 95.35%, Test Acc: 88.81%, LR: 0.001000

Experiment A: Loss Function Comparison Summary
================================================================================
          Method Final Accuracy (%)  Convergence Epoch Min Loss
    CrossEntropy              88.68                 30   0.0919
MSE_with_Softmax              88.81                 30   0.0073
================================================================================
"""

# 실험 B: 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)
def experiment_B_activation_comparison(device, results_dir):
    # 실험 설정 출력
    print("\n실험 설정:")
    print("- 데이터셋: make_moons (2D 비선형 분류)")
    print("- 네트워크: MLP [2 -> 128 -> 64 -> 2]")
    print("- 옵티마이저: Adam (lr=0.01)")
    print("- 에폭: 200")
    print("- Dead ReLU 유도: 작은 초기화 값 사용 (std=0.01)")
    
    # make_moons 데이터셋 로드
    train_loader, test_loader, input_size, num_classes = get_sklearn_data_loaders(
        'moons', n_samples=1000, batch_size=32, noise=0.2
    )
    
    # 학습 데이터 추출 (시각화용)
    X_train = []
    y_train = []
    for inputs, targets in train_loader:
        X_train.append(inputs.numpy())
        y_train.append(targets.numpy())
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    # 실험할 활성화 함수 목록
    activation_functions = ['relu', 'leaky_relu', 'sigmoid']
    results = {}
    
    # 결정 경계 시각화를 위한 subplot 구성
    fig_boundary, axes_boundary = plt.subplots(1, 3, figsize=(18, 5))
    
    # 각 활성화 함수별 실험
    for idx, activation in enumerate(activation_functions):
        print(f"\n{activation.upper()} 실험 시작...")
        
        # Dead ReLU를 유도하기 위한 작은 초기화 (ReLU만)
        init_std = 0.01 if activation == 'relu' else None
        
        # 모델 생성
        model = MLP(input_size, [128, 64], num_classes, 
                   activation=activation, init_std=init_std).to(device)
        
        # 옵티마이저 및 손실 함수 설정
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        # 모델 학습
        losses, accs, train_accs = train_model(
            model, train_loader, test_loader,
            loss_fn, optimizer, 
            epochs=200, device=device,
            verbose=True, log_interval=50
        )
        
        # Dead neuron 분석 (ReLU 계열만 수행)
        dead_ratio = 0
        if activation in ['relu', 'leaky_relu']:
            dead_ratios = analyze_dead_relu(model, train_loader, device)
            avg_dead_ratio = np.mean([ratio.mean() for ratio in dead_ratios.values()])
            dead_ratio = avg_dead_ratio
            
            print(f"\n{activation.upper()} Dead Neuron 분석:")
            for layer_name, ratio in dead_ratios.items():
                print(f"  {layer_name}: 평균 {ratio.mean():.2f}% dead neurons "
                      f"(최대: {ratio.max():.2f}%, 최소: {ratio.min():.2f}%)")
            
            # Dead neuron 히트맵 저장 (ReLU만)
            if activation == 'relu':
                visualize_dead_neurons(dead_ratios, 
                    save_path=os.path.join(results_dir, "plots", "dead_relu_heatmap.png"))
        
        # 결과 저장
        results[activation] = {
            'train_losses': losses,
            'test_accuracies': accs,
            'train_accuracies': train_accs,
            'final_accuracy': accs[-1],
            'min_loss': min(losses),
            'dead_ratio': dead_ratio,
            'convergence_epoch': next((i+1 for i, acc in enumerate(accs) if acc > 95), 200)
        }
        
        # 학습된 모델 저장
        save_model(model, os.path.join(results_dir, "models", f"model_{activation}.pth"))
        
        # 결정 경계 시각화
        ax = axes_boundary[idx]
        h = 0.02  # 격자 간격
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        model.eval()
        with torch.no_grad():
            grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
            Z = model(grid_points)
            _, Z = torch.max(Z, 1)
            Z = Z.cpu().numpy().reshape(xx.shape)
        
        # 결정 경계 및 학습 데이터 시각화
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                  cmap=plt.cm.RdYlBu, edgecolor='black', s=50)
        ax.set_title(f'{activation.upper()}\nAccuracy: {accs[-1]:.1f}%'
                    f'{f", Dead: {dead_ratio:.1f}%" if dead_ratio > 0 else ""}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    # 결정 경계 전체 시각화 저장
    plt.suptitle('Decision Boundaries by Activation Function', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "experiment_B_boundaries.png"), dpi=300)
    plt.show()
    
    # 학습 곡선 비교 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for activation in activation_functions:
        epochs = range(1, len(results[activation]['train_losses']) + 1)
        ax1.plot(epochs, results[activation]['train_losses'], 
                label=activation.upper(), linewidth=2)
        ax2.plot(epochs, results[activation]['test_accuracies'], 
                label=activation.upper(), linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss by Activation Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 로그 스케일로 학습 곡선 차이 강조
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy by Activation Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Experiment B: Activation Function Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "experiment_B_curves.png"), dpi=300)
    plt.show()
    
    # Gradient flow 분석 (각 활성화 함수별 평균 gradient)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (activation, model_results) in enumerate(results.items()):
        # 모델 재생성 (분석용)
        init_std = 0.01 if activation == 'relu' else None
        model = MLP(input_size, [128, 64], num_classes, 
                   activation=activation, init_std=init_std).to(device)
        
        # 한 배치로 gradient 계산
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        
        # gradient 크기 수집
        gradients = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad_magnitude = param.grad.abs().mean().item()
                gradients.append(grad_magnitude)
                layer_names.append(name.replace('.weight', ''))
        
        # gradient 시각화
        ax = axes[idx]
        bars = ax.bar(range(len(gradients)), gradients)
        ax.set_xticks(range(len(gradients)))
        ax.set_xticklabels(layer_names, rotation=45)
        ax.set_ylabel('Average Gradient Magnitude')
        ax.set_title(f'{activation.upper()} - Gradient Flow')
        ax.set_yscale('log')
        
        # gradient 크기에 따라 막대 색상 변경
        for i, bar in enumerate(bars):
            if gradients[i] < 1e-6:
                bar.set_color('red')
            elif gradients[i] < 1e-4:
                bar.set_color('orange')
            else:
                bar.set_color('green')
    
    plt.suptitle('Gradient Flow Analysis by Activation Function', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "experiment_B_gradients.png"), dpi=300)
    plt.show()
    
    # 결과 요약 테이블 생성 및 출력
    summary_df = create_summary_table(results, ['final_accuracy', 'convergence_epoch', 'min_loss'])
    print_summary_table(summary_df, "Experiment B: Activation Function Comparison Summary")
    
    return results  # 최종 결과 반환

# 실험 B 결과
"""
============================================================
실험 B: 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)
============================================================

실험 설정:
- 데이터셋: make_moons (2D 비선형 분류)
- 네트워크: MLP [2 -> 128 -> 64 -> 2]
- 옵티마이저: Adam (lr=0.01)
- 에폭: 200
- Dead ReLU 유도: 작은 초기화 값 사용 (std=0.01)

RELU 실험 시작...
Epoch [50/200], Loss: 0.0674, Train Acc: 96.88%, Test Acc: 97.50%, LR: 0.010000
Epoch [100/200], Loss: 0.0568, Train Acc: 97.62%, Test Acc: 98.50%, LR: 0.010000
Epoch [150/200], Loss: 0.0518, Train Acc: 97.38%, Test Acc: 98.00%, LR: 0.010000
Epoch [200/200], Loss: 0.0435, Train Acc: 98.00%, Test Acc: 98.50%, LR: 0.010000

RELU Dead Neuron 분석:
  layer_0: 평균 0.00% dead neurons (최대: 0.00%, 최소: 0.00%)
  layer_1: 평균 0.00% dead neurons (최대: 0.00%, 최소: 0.00%)

LEAKY_RELU 실험 시작...
Epoch [50/200], Loss: 0.0571, Train Acc: 97.12%, Test Acc: 97.50%, LR: 0.010000
Epoch [100/200], Loss: 0.0487, Train Acc: 97.88%, Test Acc: 96.50%, LR: 0.010000
Epoch [150/200], Loss: 0.0423, Train Acc: 98.25%, Test Acc: 98.50%, LR: 0.010000
Epoch [200/200], Loss: 0.0381, Train Acc: 98.38%, Test Acc: 97.00%, LR: 0.010000

LEAKY_RELU Dead Neuron 분석:
  layer_0: 평균 0.00% dead neurons (최대: 0.00%, 최소: 0.00%)
  layer_1: 평균 0.00% dead neurons (최대: 0.00%, 최소: 0.00%)

SIGMOID 실험 시작...
Epoch [50/200], Loss: 0.2687, Train Acc: 86.50%, Test Acc: 86.00%, LR: 0.010000
Epoch [100/200], Loss: 0.0820, Train Acc: 97.00%, Test Acc: 97.50%, LR: 0.010000
Epoch [150/200], Loss: 0.0720, Train Acc: 97.25%, Test Acc: 97.50%, LR: 0.010000
Epoch [200/200], Loss: 0.0591, Train Acc: 97.25%, Test Acc: 97.50%, LR: 0.010000

Experiment B: Activation Function Comparison Summary
================================================================================
    Method Final Accuracy (%)  Convergence Epoch Min Loss Dead Neurons (%)
      relu              98.50                  3   0.0430             0.00
leaky_relu              97.00                  2   0.0359             0.00
   sigmoid              97.50                 66   0.0513             0.00
================================================================================
"""


# 실험 C: 최적화 알고리즘 비교 (SGD vs SGD+Momentum vs Adam)
# 스케줄러 사용/미사용 모두 비교
def experiment_C_optimizer_comparison(device, results_dir):
    # 실험 설정 출력
    print("\n실험 설정:")
    print("- 데이터셋: Fashion-MNIST")
    print("- 네트워크: MLP [784 -> 256 -> 128 -> 10]")
    print("- 학습률: [0.1, 0.01, 0.001]")
    print("- 스케줄러: ExponentialLR (gamma=0.9)")
    print("- 에폭: 30")
    
    # Fashion-MNIST 데이터셋 로드
    train_loader, test_loader, input_size, num_classes = get_fashion_mnist_loaders(batch_size=128)
    
    # 사용 가능한 옵티마이저 설정
    optimizers_config = {
        'SGD': lambda params, lr: optim.SGD(params, lr=lr),
        'SGD_Momentum': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
        'Adam': lambda params, lr: optim.Adam(params, lr=lr)
    }
    
    learning_rates = [0.1, 0.01, 0.001]  # 학습률 목록
    use_scheduler_options = [False, True]  # 스케줄러 사용 여부
    
    all_results = {}  # 모든 결과 저장용
    
    # 스케줄러 사용/미사용으로 나누어 실험
    for use_scheduler in use_scheduler_options:
        scheduler_str = "with_scheduler" if use_scheduler else "without_scheduler"
        print(f"\n{'='*50}")
        print(f"스케줄러 {'사용' if use_scheduler else '미사용'} 실험")
        print(f"{'='*50}")
        
        results = {}
        
        # 각 옵티마이저-학습률 조합 실험
        for opt_name, opt_fn in optimizers_config.items():
            for lr in learning_rates:
                exp_name = f"{opt_name}_lr{lr}"
                print(f"\n{opt_name} - LR: {lr}")
                
                # 모델 생성
                model = MLP(input_size, [256, 128], num_classes).to(device)
                
                # 옵티마이저 생성
                optimizer = opt_fn(model.parameters(), lr)
                
                # 스케줄러 설정
                scheduler = None
                if use_scheduler:
                    scheduler = ExponentialLR(optimizer, gamma=0.9)
                
                # 손실 함수
                loss_fn = nn.CrossEntropyLoss()
                
                # 모델 학습
                try:
                    losses, accs, train_accs = train_model(
                        model, train_loader, test_loader,
                        loss_fn, optimizer,
                        epochs=30, scheduler=scheduler,
                        device=device, verbose=True, log_interval=10
                    )
                    
                    # 발산 여부 확인
                    if np.isnan(losses[-1]) or losses[-1] > 10:
                        print(f"  [경고] {exp_name} 발산!")
                        results[exp_name] = {
                            'train_losses': losses,
                            'test_accuracies': accs,
                            'train_accuracies': train_accs,
                            'final_accuracy': 0,
                            'convergence_speed': 30,
                            'diverged': True
                        }
                    else:
                        results[exp_name] = {
                            'train_losses': losses,
                            'test_accuracies': accs,
                            'train_accuracies': train_accs,
                            'final_accuracy': accs[-1],
                            'convergence_speed': next((i+1 for i, acc in enumerate(accs) if acc > 85), 30),
                            'diverged': False
                        }
                    
                    # 학습된 모델 저장
                    save_model(model, os.path.join(results_dir, "models", 
                              f"model_{opt_name}_lr{lr}_{scheduler_str}.pth"))
                    
                except Exception as e:
                    # 예외 처리
                    print(f"  [에러] {exp_name}: {str(e)}")
                    results[exp_name] = {
                        'train_losses': [float('inf')],
                        'test_accuracies': [0],
                        'train_accuracies': [0],
                        'final_accuracy': 0,
                        'convergence_speed': 30,
                        'diverged': True
                    }
        
        # 현재 실험 결과 저장
        all_results[scheduler_str] = results
        
        # 시각화: 학습률별 비교 (Loss/Accuracy)
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        for i, lr in enumerate(learning_rates):
            ax_loss = axes[i, 0]
            ax_acc = axes[i, 1]
            
            for opt_name in optimizers_config.keys():
                exp_name = f"{opt_name}_lr{lr}"
                if exp_name in results and not results[exp_name].get('diverged', False):
                    epochs = range(1, len(results[exp_name]['train_losses']) + 1)
                    
                    ax_loss.plot(epochs, results[exp_name]['train_losses'], 
                               label=opt_name, linewidth=2)
                    ax_acc.plot(epochs, results[exp_name]['test_accuracies'], 
                              label=opt_name, linewidth=2)
            
            ax_loss.set_title(f'Loss (LR={lr})')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            if lr == 0.1:  # 높은 학습률은 로그 스케일
                ax_loss.set_yscale('log')
            
            ax_acc.set_title(f'Accuracy (LR={lr})')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy (%)')
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
        
        plt.suptitle(f'Optimizer Comparison - {scheduler_str.replace("_", " ").title()}', 
                    fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plots", 
                   f"experiment_C_{scheduler_str}.png"), dpi=300)
        plt.show()
    
    # 종합 비교: 스케줄러 효과
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    for i, opt_name in enumerate(optimizers_config.keys()):
        for j, lr in enumerate(learning_rates):
            ax = axes[i, j]
            
            exp_name = f"{opt_name}_lr{lr}"
            
            # 스케줄러 사용/미사용 비교
            for scheduler_use, results in all_results.items():
                if exp_name in results and not results[exp_name].get('diverged', False):
                    epochs = range(1, len(results[exp_name]['test_accuracies']) + 1)
                    label = 'With Scheduler' if 'with' in scheduler_use else 'Without Scheduler'
                    linestyle = '-' if 'with' in scheduler_use else '--'
                    
                    ax.plot(epochs, results[exp_name]['test_accuracies'], 
                           label=label, linewidth=2, linestyle=linestyle)
            
            ax.set_title(f'{opt_name} (LR={lr})')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Test Accuracy (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Scheduler Effect on Different Optimizers', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "experiment_C_scheduler_effect.png"), dpi=300)
    plt.show()
    
    # 최종 결과 요약 출력
    print("\n실험 C 종합 결과:")
    print("="*100)
    print(f"{'Optimizer':<15} {'LR':<10} {'Scheduler':<15} {'Final Acc (%)':<15} "
          f"{'Conv. Epoch':<15} {'Status':<10}")
    print("-"*100)
    
    for scheduler_use, results in all_results.items():
        for opt_name in optimizers_config.keys():
            for lr in learning_rates:
                exp_name = f"{opt_name}_lr{lr}"
                if exp_name in results:
                    scheduler_status = 'Yes' if 'with' in scheduler_use else 'No'
                    status = 'Diverged' if results[exp_name].get('diverged', False) else 'OK'
                    
                    print(f"{opt_name:<15} {lr:<10} {scheduler_status:<15} "
                          f"{results[exp_name]['final_accuracy']:<15.2f} "
                          f"{results[exp_name]['convergence_speed']:<15} {status:<10}")
    
    print("="*100)
    
    # 최고 성능 조합 찾기
    best_config = None
    best_accuracy = 0
    
    for scheduler_use, results in all_results.items():
        for exp_name, metrics in results.items():
            if not metrics.get('diverged', False) and metrics['final_accuracy'] > best_accuracy:
                best_accuracy = metrics['final_accuracy']
                best_config = (exp_name, scheduler_use)
    
    if best_config:
        print(f"\n최고 성능 조합: {best_config[0]} "
              f"({'with scheduler' if 'with' in best_config[1] else 'without scheduler'}) "
              f"- 정확도: {best_accuracy:.2f}%")
    
    return all_results  # 최종 결과 반환

# 실험 C 결과
"""
============================================================
실험 C: 최적화 알고리즘 비교 (SGD vs SGD+Momentum vs Adam)
============================================================

실험 설정:
- 데이터셋: Fashion-MNIST
- 네트워크: MLP [784 -> 256 -> 128 -> 10]
- 학습률: [0.1, 0.01, 0.001]
- 스케줄러: ExponentialLR (gamma=0.9)
- 에폭: 30

==================================================
스케줄러 미사용 실험
==================================================

SGD - LR: 0.1
Epoch [10/30], Loss: 0.2587, Train Acc: 90.54%, Test Acc: 87.50%, LR: 0.100000
Epoch [20/30], Loss: 0.1820, Train Acc: 93.32%, Test Acc: 88.02%, LR: 0.100000
Epoch [30/30], Loss: 0.1306, Train Acc: 95.25%, Test Acc: 88.76%, LR: 0.100000

SGD - LR: 0.01
Epoch [10/30], Loss: 0.3864, Train Acc: 86.26%, Test Acc: 84.60%, LR: 0.010000
Epoch [20/30], Loss: 0.3360, Train Acc: 88.02%, Test Acc: 85.87%, LR: 0.010000
Epoch [30/30], Loss: 0.3062, Train Acc: 89.11%, Test Acc: 86.60%, LR: 0.010000

SGD - LR: 0.001
Epoch [10/30], Loss: 0.6130, Train Acc: 78.68%, Test Acc: 77.83%, LR: 0.001000
Epoch [20/30], Loss: 0.5178, Train Acc: 81.93%, Test Acc: 80.77%, LR: 0.001000
Epoch [30/30], Loss: 0.4752, Train Acc: 83.30%, Test Acc: 82.17%, LR: 0.001000

SGD_Momentum - LR: 0.1
Epoch [10/30], Loss: 0.2809, Train Acc: 89.60%, Test Acc: 87.39%, LR: 0.100000
Epoch [20/30], Loss: 0.2190, Train Acc: 91.77%, Test Acc: 87.11%, LR: 0.100000
Epoch [30/30], Loss: 0.1892, Train Acc: 93.04%, Test Acc: 87.60%, LR: 0.100000

SGD_Momentum - LR: 0.01
Epoch [10/30], Loss: 0.2546, Train Acc: 90.66%, Test Acc: 87.60%, LR: 0.010000
Epoch [20/30], Loss: 0.1791, Train Acc: 93.51%, Test Acc: 88.30%, LR: 0.010000
Epoch [30/30], Loss: 0.1326, Train Acc: 95.20%, Test Acc: 89.17%, LR: 0.010000

SGD_Momentum - LR: 0.001
Epoch [10/30], Loss: 0.3883, Train Acc: 86.19%, Test Acc: 85.10%, LR: 0.001000
Epoch [20/30], Loss: 0.3374, Train Acc: 87.94%, Test Acc: 85.86%, LR: 0.001000
Epoch [30/30], Loss: 0.3072, Train Acc: 89.06%, Test Acc: 86.68%, LR: 0.001000

Adam - LR: 0.1
Epoch [10/30], Loss: 2.3088, Train Acc: 10.21%, Test Acc: 10.00%, LR: 0.100000
Epoch [20/30], Loss: 2.3090, Train Acc: 9.81%, Test Acc: 10.00%, LR: 0.100000
Epoch [30/30], Loss: 2.3094, Train Acc: 10.32%, Test Acc: 10.00%, LR: 0.100000

Adam - LR: 0.01
Epoch [10/30], Loss: 0.3203, Train Acc: 88.55%, Test Acc: 86.21%, LR: 0.010000
Epoch [20/30], Loss: 2.3090, Train Acc: 9.81%, Test Acc: 10.00%, LR: 0.100000
Epoch [30/30], Loss: 2.3094, Train Acc: 10.32%, Test Acc: 10.00%, LR: 0.100000

Adam - LR: 0.01
Epoch [10/30], Loss: 0.3203, Train Acc: 88.55%, Test Acc: 86.21%, LR: 0.010000
Epoch [30/30], Loss: 2.3094, Train Acc: 10.32%, Test Acc: 10.00%, LR: 0.100000

Adam - LR: 0.01
Epoch [10/30], Loss: 0.3203, Train Acc: 88.55%, Test Acc: 86.21%, LR: 0.010000

Adam - LR: 0.01
Epoch [10/30], Loss: 0.3203, Train Acc: 88.55%, Test Acc: 86.21%, LR: 0.010000
Epoch [10/30], Loss: 0.3203, Train Acc: 88.55%, Test Acc: 86.21%, LR: 0.010000
Epoch [20/30], Loss: 0.2920, Train Acc: 89.75%, Test Acc: 87.09%, LR: 0.010000
Epoch [20/30], Loss: 0.2920, Train Acc: 89.75%, Test Acc: 87.09%, LR: 0.010000
Epoch [30/30], Loss: 0.2683, Train Acc: 90.44%, Test Acc: 86.20%, LR: 0.010000

Adam - LR: 0.001
Epoch [10/30], Loss: 0.2177, Train Acc: 91.83%, Test Acc: 88.22%, LR: 0.001000
Epoch [20/30], Loss: 0.1371, Train Acc: 94.79%, Test Acc: 88.80%, LR: 0.001000
Epoch [30/30], Loss: 0.0889, Train Acc: 96.59%, Test Acc: 89.09%, LR: 0.001000

==================================================
스케줄러 사용 실험
==================================================

SGD - LR: 0.1
Epoch [10/30], Loss: 0.2631, Train Acc: 90.39%, Test Acc: 87.67%, LR: 0.034868
Epoch [20/30], Loss: 0.2157, Train Acc: 92.31%, Test Acc: 88.31%, LR: 0.012158
Epoch [30/30], Loss: 0.1998, Train Acc: 92.95%, Test Acc: 88.73%, LR: 0.004239

SGD - LR: 0.01
Epoch [10/30], Loss: 0.4139, Train Acc: 85.43%, Test Acc: 83.82%, LR: 0.003487
Epoch [20/30], Loss: 0.3888, Train Acc: 86.25%, Test Acc: 84.63%, LR: 0.001216
Epoch [30/30], Loss: 0.3815, Train Acc: 86.49%, Test Acc: 84.77%, LR: 0.000424

SGD - LR: 0.001
Epoch [10/30], Loss: 0.6881, Train Acc: 76.56%, Test Acc: 75.87%, LR: 0.000349
Epoch [20/30], Loss: 0.6273, Train Acc: 78.34%, Test Acc: 77.51%, LR: 0.000122
Epoch [30/30], Loss: 0.6122, Train Acc: 78.83%, Test Acc: 77.94%, LR: 0.000042

SGD_Momentum - LR: 0.1
Epoch [10/30], Loss: 0.2163, Train Acc: 91.70%, Test Acc: 88.35%, LR: 0.034868
Epoch [20/30], Loss: 0.1194, Train Acc: 95.48%, Test Acc: 89.29%, LR: 0.012158
Epoch [30/30], Loss: 0.0776, Train Acc: 97.37%, Test Acc: 89.30%, LR: 0.004239

SGD_Momentum - LR: 0.01
Epoch [10/30], Loss: 0.2649, Train Acc: 90.38%, Test Acc: 87.98%, LR: 0.003487
Epoch [20/30], Loss: 0.2195, Train Acc: 92.16%, Test Acc: 88.46%, LR: 0.001216
Epoch [30/30], Loss: 0.2040, Train Acc: 92.77%, Test Acc: 88.61%, LR: 0.000424

SGD_Momentum - LR: 0.001
Epoch [10/30], Loss: 0.4139, Train Acc: 85.32%, Test Acc: 83.94%, LR: 0.000349
Epoch [20/30], Loss: 0.3884, Train Acc: 86.25%, Test Acc: 84.79%, LR: 0.000122
Epoch [30/30], Loss: 0.3811, Train Acc: 86.52%, Test Acc: 84.87%, LR: 0.000042

Adam - LR: 0.1
Epoch [10/30], Loss: 2.3053, Train Acc: 9.92%, Test Acc: 10.00%, LR: 0.034868
Epoch [20/30], Loss: 2.3035, Train Acc: 10.00%, Test Acc: 10.00%, LR: 0.012158
Epoch [30/30], Loss: 2.3029, Train Acc: 10.04%, Test Acc: 10.00%, LR: 0.004239

Adam - LR: 0.01
Epoch [10/30], Loss: 0.2678, Train Acc: 89.91%, Test Acc: 87.44%, LR: 0.003487
Epoch [20/30], Loss: 0.1891, Train Acc: 92.86%, Test Acc: 88.13%, LR: 0.001216
Epoch [30/30], Loss: 0.1547, Train Acc: 94.21%, Test Acc: 88.10%, LR: 0.000424

Adam - LR: 0.001
Epoch [10/30], Loss: 0.1933, Train Acc: 92.98%, Test Acc: 88.71%, LR: 0.000349
Epoch [20/30], Loss: 0.1164, Train Acc: 95.98%, Test Acc: 89.81%, LR: 0.000122
Epoch [30/30], Loss: 0.0877, Train Acc: 97.22%, Test Acc: 89.81%, LR: 0.000042

실험 C 종합 결과:
====================================================================================================
Optimizer       LR         Scheduler       Final Acc (%)   Conv. Epoch     Status
----------------------------------------------------------------------------------------------------
SGD             0.1        Yes             88.76           3               OK
SGD             0.01       Yes             86.60           13              OK
SGD             0.001      Yes             82.17           30              OK
SGD_Momentum    0.1        Yes             87.60           3               OK
SGD_Momentum    0.01       Yes             89.17           2               OK
SGD_Momentum    0.001      Yes             86.68           10              OK
Adam            0.1        Yes             10.00           30              OK
Adam            0.01       Yes             86.20           4               OK
Adam            0.001      Yes             89.09           2               OK
SGD             0.1        Yes             88.73           4               OK
SGD             0.01       Yes             84.77           30              OK
SGD             0.001      Yes             77.94           30              OK
SGD_Momentum    0.1        Yes             89.30           3               OK
SGD_Momentum    0.01       Yes             88.61           3               OK
SGD_Momentum    0.001      Yes             84.87           30              OK
Adam            0.1        Yes             10.00           30              OK
Adam            0.01       Yes             88.10           5               OK
Adam            0.001      Yes             89.81           2               OK
====================================================================================================

최고 성능 조합: Adam_lr0.001 (with scheduler) - 정확도: 89.81%
"""