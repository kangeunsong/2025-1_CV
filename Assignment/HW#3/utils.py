"""
유틸리티 함수들
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json
from datetime import datetime
import pandas as pd

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_results_directory():
    """결과 저장 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "analysis"), exist_ok=True)
    return results_dir

def save_results(results, filename):
    """결과를 pickle 파일로 저장"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

def load_results(filename):
    """pickle 파일에서 결과 로드"""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results

def save_model(model, filepath):
    """모델 저장"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.layers[0].in_features,
            'hidden_sizes': [layer.out_features for layer in model.layers],
            'num_classes': model.output_layer.out_features,
            'activation': model.activation_name if hasattr(model, 'activation_name') else 'relu'
        }
    }, filepath)

def load_model(model_class, filepath, device):
    """모델 로드"""
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['model_config']
    model = model_class(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)

# 데이터 로더 함수들
def get_fashion_mnist_loaders(batch_size=128, validation_split=0.1):
    """Fashion-MNIST 데이터 로더"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 데이터셋 다운로드
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Validation set 분리
    if validation_split > 0:
        val_size = int(len(train_dataset) * validation_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader, 784, 10
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader, 784, 10

def get_sklearn_data_loaders(dataset_type='moons', n_samples=1000, batch_size=32, noise=0.2):
    """Scikit-learn 2D 데이터셋 로더"""
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset_type == 'blobs':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                  n_informative=2, n_clusters_per_class=1, n_classes=2,
                                  random_state=42)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train/Test 분할 (80/20)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Tensor로 변환
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, 2, 2

# 학습 함수
def train_model(model, train_loader, test_loader, loss_fn, optimizer, 
                epochs=30, scheduler=None, device='cpu', use_softmax_mse=False,
                verbose=True, log_interval=5):
    """
    모델 학습 함수
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        test_loader: 테스트 데이터 로더
        loss_fn: 손실 함수
        optimizer: 옵티마이저
        epochs: 학습 에폭 수
        scheduler: 학습률 스케줄러
        device: 학습 디바이스
        use_softmax_mse: MSE loss 사용 시 softmax 적용 여부
        verbose: 학습 과정 출력 여부
        log_interval: 로그 출력 간격
    
    Returns:
        train_losses: 에폭별 학습 손실
        test_accuracies: 에폭별 테스트 정확도
        train_accuracies: 에폭별 학습 정확도
    """
    model.to(device)
    
    train_losses = []
    test_accuracies = []
    train_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss 계산
            if use_softmax_mse:
                # MSE Loss를 위한 특별 처리
                outputs_softmax = torch.softmax(outputs, dim=1)
                targets_onehot = torch.zeros_like(outputs_softmax)
                targets_onehot.scatter_(1, targets.view(-1, 1), 1)
                loss = loss_fn(outputs_softmax, targets_onehot)
            else:
                loss = loss_fn(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping (선택적)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 통계 업데이트
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Evaluation phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
        
        # 에폭 통계 저장
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        # 로그 출력
        if verbose and (epoch + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Test Acc: {test_accuracy:.2f}%, '
                  f'LR: {current_lr:.6f}')
    
    return train_losses, test_accuracies, train_accuracies

# 시각화 함수들
def plot_learning_curves(train_losses, test_accuracies, train_accuracies=None,
                        title="Learning Curves", save_path=None):
    """학습 곡선 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 곡선
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy 곡선
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    if train_accuracies is not None:
        ax2.plot(epochs, train_accuracies, 'b--', label='Train Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison(results_dict, metric='accuracy', title="Comparison", save_path=None):
    """여러 실험 결과 비교 플롯"""
    plt.figure(figsize=(12, 6))
    
    for name, results in results_dict.items():
        if metric == 'accuracy':
            data = results.get('test_accuracies', results.get('accuracies', []))
        elif metric == 'loss':
            data = results.get('train_losses', results.get('losses', []))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        epochs = range(1, len(data) + 1)
        plt.plot(epochs, data, label=name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(results_dict, metrics=['final_accuracy', 'convergence_epoch', 'min_loss']):
    """실험 결과 요약 테이블 생성"""
    summary_data = []
    
    for name, results in results_dict.items():
        row = {'Method': name}
        
        if 'final_accuracy' in metrics:
            accs = results.get('test_accuracies', results.get('accuracies', []))
            row['Final Accuracy (%)'] = f"{accs[-1]:.2f}" if accs else "N/A"
        
        if 'convergence_epoch' in metrics:
            accs = results.get('test_accuracies', results.get('accuracies', []))
            threshold = 90  # 90% 정확도를 수렴 기준으로
            conv_epoch = next((i+1 for i, acc in enumerate(accs) if acc > threshold), len(accs))
            row['Convergence Epoch'] = conv_epoch
        
        if 'min_loss' in metrics:
            losses = results.get('train_losses', results.get('losses', []))
            row['Min Loss'] = f"{min(losses):.4f}" if losses else "N/A"
        
        if 'dead_ratio' in results:
            row['Dead Neurons (%)'] = f"{results['dead_ratio']:.2f}"
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def print_summary_table(df, title="Summary"):
    """요약 테이블 출력"""
    print(f"\n{title}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

# 2D 데이터 시각화
def plot_decision_boundary(model, X, y, title="Decision Boundary", save_path=None):
    """2D 데이터에 대한 결정 경계 시각화"""
    device = next(model.parameters()).device
    
    # 메시그리드 생성
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 예측
    model.eval()
    with torch.no_grad():
        grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
        Z = model(grid_points)
        _, Z = torch.max(Z, 1)
        Z = Z.cpu().numpy().reshape(xx.shape)
    
    # 플롯
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_metrics(outputs, targets):
    """정확도, 정밀도, 재현율, F1 점수 계산"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    
    # 이진 분류의 경우 추가 메트릭
    if len(torch.unique(targets)) == 2:
        tp = ((predicted == 1) & (targets == 1)).sum().item()
        fp = ((predicted == 1) & (targets == 0)).sum().item()
        fn = ((predicted == 0) & (targets == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    else:
        return {'accuracy': accuracy}