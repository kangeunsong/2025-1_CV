# 메인 실행 파일

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_moons, make_circles
import matplotlib
matplotlib.use('Agg')  # matplotlib 백엔드를 'Agg'로 설정 (GUI 창 없이 파일로 저장)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import warnings
warnings.filterwarnings('ignore')  # 경고 메시지 무시

# 사용자 정의 모듈 임포트
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

# 시드 고정 함수 (재현성을 위한 시드 고정)
def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 명령줄 인자 파싱 (실행할 실험 목록 반환)
def parse_arguments():
    if len(sys.argv) == 1:
        # 인자가 없으면 모든 실험 실행
        return ['A', 'B', 'C']
    
    valid_experiments = ['A', 'B', 'C']
    requested_experiments = []
    
    for arg in sys.argv[1:]:
        arg_upper = arg.upper()
        if arg_upper in valid_experiments:
            if arg_upper not in requested_experiments:
                requested_experiments.append(arg_upper)
        else:
            print(f"경고: '{arg}'는 유효하지 않은 실험입니다. A, B, C 중 선택하세요.")
    
    if not requested_experiments:
        # 유효하지 않은 인자만 있으면 실행 종료
        print("유효한 실험이 지정되지 않았습니다. 사용법:")
        print("  python main.py          # 모든 실험 실행")
        print("  python main.py A        # 실험 A만 실행")
        print("  python main.py B        # 실험 B만 실행")
        print("  python main.py C        # 실험 C만 실행")
        print("  python main.py A C      # 실험 A와 C 실행")
        sys.exit(1)
    
    return requested_experiments

# 메인 함수
def main():
    # 실행할 실험 결정
    experiments_to_run = parse_arguments()
    
    print("="*60)
    print("2025-1 Computer Vision [HW#3] MLP 실험")
    print(f"실행할 실험: {', '.join(experiments_to_run)}")
    print("="*60)
    
    # 시드 고정
    set_random_seeds(42)
    
    # 결과 저장 디렉토리 생성
    results_dir = create_results_directory()
    
    all_results = {}  # 모든 실험 결과 저장
    
    # 실험 A: 손실 함수 비교
    if 'A' in experiments_to_run:
        print("\n" + "="*60)
        print("실험 A: 손실 함수 비교 (CrossEntropy vs MSE)")
        print("="*60)
        results_A = experiment_A_loss_comparison(device, results_dir)
        all_results['experiment_A'] = results_A
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 실험 B: 활성화 함수 비교
    if 'B' in experiments_to_run:
        print("\n" + "="*60)
        print("실험 B: 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)")
        print("="*60)
        results_B = experiment_B_activation_comparison(device, results_dir)
        all_results['experiment_B'] = results_B
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 실험 C: 최적화 알고리즘 비교
    if 'C' in experiments_to_run:
        print("\n" + "="*60)
        print("실험 C: 최적화 알고리즘 비교 (SGD vs SGD+Momentum vs Adam)")
        print("="*60)
        results_C = experiment_C_optimizer_comparison(device, results_dir)
        all_results['experiment_C'] = results_C
    
    # 전체 결과 저장 및 최종 보고서 생성
    if all_results:
        save_results(all_results, os.path.join(results_dir, 'all_experiment_results.pkl'))
        generate_experiment_report(all_results, results_dir)
        
        print(f"\n실행한 실험이 완료되었습니다!")
        print(f"결과는 {results_dir} 디렉토리에 저장되었습니다.")
    else:
        print("\n실행된 실험이 없습니다.")
        # 빈 디렉토리 삭제
        import shutil
        shutil.rmtree(results_dir)

# 메인 함수 호출
if __name__ == "__main__":
    main()