import os
os.system('clear')  # Linux나 macOS    

import numpy as np

# from hdbo.boattack.models.additive_gp_decomp import Additive_GPModel_Learn_Decomp
try:
    from hdbo.boattack.models.additive_gp_decomp import Additive_GPModel_Learn_Decomp
    from hdbo.boattack.models.base import BaseModel
except ImportError as e:
    print(f"ImportError: {e}")

# 예제 데이터 생성
np.random.seed(42)
X_train = np.random.rand(100, 5)  # 100개의 5차원 입력 데이터
Y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, (100,))

# print(X_train)

# 모델 초기화
model = Additive_GPModel_Learn_Decomp(
    n_subspaces=3,
    update_freq=10,
    normalize_Y=True,
    ARD=True,
    exact_feval=False,
    optimize_restarts=2
)

# 모델 학습
for i in range(50):
    model._update_model(X_train, Y_train, itr=i)

# 새로운 데이터에 대한 예측
X_test = np.random.rand(10, 5)  # 10개의 새로운 테스트 데이터

# 전체 예측
predictions = []
for x in X_test:
    m_total = 0
    v_total = 0
    for subspace_id in range(model.n_sub):
        m_sub, s_sub = model.predictSub(x, subspace_id)
        m_total += m_sub
        v_total += s_sub**2
    predictions.append((m_total[0, 0], np.sqrt(v_total[0, 0])))

print("Predictions (mean, std):")
for i, (mean, std) in enumerate(predictions):
    print(f"Sample {i+1}: Mean = {mean:.4f}, Std = {std:.4f}")

# 특정 부분공간에 대한 예측 및 그래디언트
subspace_id = 0  # 첫 번째 부분공간에 대해 예측
m_sub, s_sub, dmdx_sub, dsdx_sub = model.predictSub_withGradients(X_test[0], subspace_id)

print(f"\nPrediction for subspace {subspace_id}:")
print(f"Mean: {m_sub[0, 0]:.4f}")
print(f"Std: {s_sub[0, 0]:.4f}")
print(f"Gradient of mean: {dmdx_sub[0]}")
print(f"Gradient of std: {dsdx_sub[0]}")