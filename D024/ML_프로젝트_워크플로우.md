# ML 프로젝트 범용 워크플로우

주제가 무엇이든 적용 가능한 단계별 흐름 (분류 기준, 회귀는 일부만 다름)

---

## 0단계. 문제 정의
- 타겟 `y`가 **숫자**(회귀)인지 **범주**(분류)인지 확인
- `y`가 원본 데이터에 이미 있는지, 파생시켜야 하는지 확인

---

## 1단계. 데이터 분리 (제일 먼저, 딱 한 번)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 분류일 때만 stratify
)
```

- `X_test`, `y_test`는 이 순간부터 끝까지 절대 안 봄 (맨 마지막에 1번만 사용)

---

## 2단계. 베이스라인 먼저

```python
from sklearn.dummy import DummyClassifier  # 회귀면 DummyRegressor

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy.score(X_test, y_test)  # 이 숫자보다 못하면 모델 의미 없음
```

- "아무 생각 없이 찍었을 때" 기준선 확보 → 나중에 모델 성능과 비교할 잣대

---

## 3단계. 전처리 + 모델을 Pipeline으로 묶기

```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),      # 결측치
    ('scaler', StandardScaler()),                        # 스케일링 (트리 계열이면 생략 가능)
    ('model', RandomForestClassifier(random_state=42))   # 일단 랜덤포레스트
])
```

- 여기서 데이터 누수 걱정 끝 (fold마다 자동으로 train만 fit)
- **랜덤포레스트를 첫 모델로 쓰는 이유**: 기본값도 성능 잘 나옴, 스케일링 거의 불필요, 과적합에 강함

---

## 4단계. 교차검증으로 진짜 성능 확인

```python
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 분류면 Stratified
scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['accuracy', 'f1'])
scores['test_accuracy'].mean(), scores['test_accuracy'].std()
```

- 평균 = 성능, 표준편차 = 안정성 (표준편차 크면 데이터/분할 민감도 의심)
- 이 값을 2단계 베이스라인과 비교

---

## 5단계. 진단

| 신호 | 판단 |
|---|---|
| CV 점수가 베이스라인과 비슷 | 피처가 정보 없음 → **1번 레버(피처)로 돌아가기** |
| train↑ test↓ 격차 큼 | 과적합 → 규제/`max_depth` 조정 또는 데이터 추가 |
| CV 점수가 기대보다 훨씬 높음 | 데이터 누수 의심 → Pipeline 밖 전처리 없었는지 재점검 |

---

## 6단계. 성능 부족하면 레버 순서대로

**순서: 피처 → 모델 → 튜닝** (역순으로 하면 나중에 헛수고됨)

1. **피처** — 파생변수 추가/제거, 인코딩 재검토, 누수 피처 의심 점검
2. **모델** — RandomForest → XGBoost/GBM으로 교체 시도
3. **튜닝** — 그리드/랜덤서치로 하이퍼파라미터 탐색 (**피처 확정된 후에만!**)

```python
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None]
}
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1')
grid.fit(X_train, y_train)
```

---

## 7단계. 필요시 해석

- `feature_importances_` (불순도 기반) 또는 `permutation_importance` (더 신뢰도 높음)로 어떤 피처가 예측을 이끄는지 확인
- 필요하면 SHAP으로 개별 예측 설명

---

## 8단계. 최종 확인 (test 20%, 딱 1번)

```python
final_score = grid.best_estimator_.score(X_test, y_test)
```

- 여기서 나온 점수가 **진짜 최종 성능** — 이거 보고 또 모델 고치면 안 됨

---

## 9단계. 저장

```python
import joblib
joblib.dump(grid.best_estimator_, 'model.pkl')  # Pipeline 전체 통째로 저장
```

- 재로드 후 예측이 일치하는지 확인

---

## 한 줄 흐름

나누기(1) → 베이스라인(2) → Pipeline 만들기(3) → CV로 성능 재기(4) → 진단(5) → 안되면 피처>모델>튜닝 순으로 개선(6) → 해석(7) → test로 최종 확인(8) → 저장(9)
