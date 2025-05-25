# MLP 모델 정의

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 다층 퍼셉트론 (Multi-Layer Perceptron) 모델

# input_size: 입력 차원
# hidden_sizes: 은닉층 크기 리스트
# num_classes: 출력 클래스 수
# activation: 활성화 함수 ('relu', 'leaky_relu', 'sigmoid')
# init_std: 가중치 초기화 표준편차 (None이면 기본값 사용)
# use_batch_norm: 배치 정규화 사용 여부
# dropout_rate: 드롭아웃 비율

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128], num_classes=10,
                 activation='relu', init_std=None, use_batch_norm=False, 
                 dropout_rate=0.0):
        super(MLP, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # 레이어 구성
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        self.dropout_layers = nn.ModuleList() if dropout_rate > 0 else None
        
        prev_size = input_size
        
        # 은닉층 구성
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            
            if dropout_rate > 0:
                self.dropout_layers.append(nn.Dropout(dropout_rate))
                
            prev_size = hidden_size
        
        # 출력층
        self.output_layer = nn.Linear(prev_size, num_classes)
        
        # 활성화 함수 설정
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # 가중치 초기화
        if init_std is not None:
            self.apply(lambda m: self._init_weights(m, init_std))
        else:
            self.apply(self._init_weights_default)
    
    # 사용자 정의 가중치 초기화
    def _init_weights(self, m, std):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    # 기본 가중치 초기화 (He or Xavier)
    def _init_weights_default(self, m):
        if isinstance(m, nn.Linear):
            if self.activation_name in ['relu', 'leaky_relu']:
                # He 초기화 (ReLU 계열)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                # Xavier 초기화 (Sigmoid, Tanh)
                nn.init.xavier_normal_(m.weight)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    # 순전파
    def forward(self, x):
        # 입력을 1D로 평탄화
        x = x.view(x.size(0), -1)
        
        # 은닉층 통과
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            
            x = self.activation(x)
            
            if self.dropout_rate > 0 and self.training:
                x = self.dropout_layers[i](x)
        
        # 출력층
        x = self.output_layer(x)
        return x
    
    # 특정 레이어의 활성화 값 반환
    def get_activations(self, x, layer_idx=None):
        x = x.view(x.size(0), -1)
        activations = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            
            if layer_idx is None or i == layer_idx:
                activations.append(x.detach().cpu())
            
            if self.dropout_rate > 0 and self.training:
                x = self.dropout_layers[i](x)
        
        return activations if layer_idx is None else activations[0]
    
    # 모델의 전체 파라미터 수 계산
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    # 각 레이어의 가중치 반환
    def get_layer_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.detach().cpu().numpy())
        weights.append(self.output_layer.weight.detach().cpu().numpy())
        return weights
    
    # 각 레이어의 그래디언트 반환
    def get_layer_gradients(self):
        gradients = []
        for layer in self.layers:
            if layer.weight.grad is not None:
                gradients.append(layer.weight.grad.detach().cpu().numpy())
        if self.output_layer.weight.grad is not None:
            gradients.append(self.output_layer.weight.grad.detach().cpu().numpy())
        return gradients

# Skip Connection이 있는 MLP (추가 실험용)
class MLPWithSkipConnection(MLP):
    def __init__(self, input_size, hidden_sizes=[256, 128], num_classes=10,
                 activation='relu', init_std=None):
        super().__init__(input_size, hidden_sizes, num_classes, activation, init_std)
        
        # Skip connection을 위한 추가 레이어
        if len(hidden_sizes) > 1:
            self.skip_connection = nn.Linear(hidden_sizes[0], hidden_sizes[-1])
        else:
            self.skip_connection = None
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 첫 번째 은닉층
        x = self.layers[0](x)
        x = self.activation(x)
        first_hidden = x.clone()
        
        # 나머지 은닉층
        for i in range(1, len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)
            
            # Skip connection (마지막 은닉층에서)
            if i == len(self.layers) - 1 and self.skip_connection is not None:
                x = x + self.skip_connection(first_hidden)
                x = self.activation(x)
        
        # 출력층
        x = self.output_layer(x)
        return x

# 모델 생성 팩토리 함수
def create_model(model_type='standard', **kwargs):
    if model_type == 'standard':
        return MLP(**kwargs)
    elif model_type == 'skip':
        return MLPWithSkipConnection(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")