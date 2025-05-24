#!/usr/bin/env python3
"""
컴퓨터 비전 과제 3: MLP 실험
메인 실행 파일
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# 모듈 임포트
from models import MLP
from utils import (
    get_fashion_mnist_loaders,
    get_sklearn_data_loaders,
    train_model,
    save_results,
    create_results_directory
)
from experiments import (
    experiment_A_loss_comparison,
    experiment_B_activation_comparison,
    experiment_C_optimizer_comparison
)
from analysis import (
    analyze_dead_relu,
    visualize_dead_neurons,
    analyze_gradient_flow,
    visualize_activation_distribution,
    generate_experiment_report
)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Random seed 고정
def set_random_seeds(seed=42):
    """재현성을 위한 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """메인 실행 함수"""
    print("="*60)
    print("컴퓨터 비전 과제 3: MLP 실험")
    print("="*60)
    
    # 시드 고정
    set_random_seeds(42)
    
    # 결과 저장 디렉토리 생성
    results_dir = create_results_directory()
    
    # 전체 실험 결과 저장용
    all_results = {}
    
    # 실험 A: 손실 함수 비교
    print("\n" + "="*60)
    print("실험 A: 손실 함수 비교 (CrossEntropy vs MSE)")
    print("="*60)
    results_A = experiment_A_loss_comparison(device, results_dir)
    all_results['experiment_A'] = results_A
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 실험 B: 활성화 함수 비교
    print("\n" + "="*60)
    print("실험 B: 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)")
    print("="*60)
    results_B = experiment_B_activation_comparison(device, results_dir)
    all_results['experiment_B'] = results_B
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 실험 C: 최적화 알고리즘 비교
    print("\n" + "="*60)
    print("실험 C: 최적화 알고리즘 비교 (SGD vs SGD+Momentum vs Adam)")
    print("="*60)
    results_C = experiment_C_optimizer_comparison(device, results_dir)
    all_results['experiment_C'] = results_C
    
    # 전체 결과 저장
    save_results(all_results, os.path.join(results_dir, 'all_experiment_results.pkl'))
    
    # 최종 보고서 생성
    print("\n" + "="*60)
    print("최종 분석 보고서 생성 중...")
    print("="*60)
    generate_experiment_report(all_results, results_dir)
    
    print(f"\n모든 실험이 완료되었습니다!")
    print(f"결과는 {results_dir} 디렉토리에 저장되었습니다.")
    print("\n제출 체크리스트:")
    print("1. GitHub에 코드 업로드")
    print("2. README.md 작성 완료")
    print("3. 보고서 PDF 생성")
    print("4. 모든 그래프 파일 확인")

if __name__ == "__main__":
    main()