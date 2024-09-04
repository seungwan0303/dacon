# 암환자 유전체 데이터 기반 암종 분류 AI 모델 개발

# Import library

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb

# Load Data

train = pd.read_csv("data/images/train.csv")
test = pd.read_csv("data/images/test.csv")

# Data Preprocessing

# SUBCLASS 가 범주형이기 때문에 LabelEncoder 사용'

le_subclass = LabelEncoder()
train['SUBCLASS'] = le_subclass.fit_transform(train['SUBCLASS'])

# 변환된 레이블 확인

for i, label in enumerate(le_subclass.classes_):
    print(f"원래 레이블: {label}, 변환된 숫자: {i}")

X = train.drop(columns=['SUBCLASS', 'ID'])
y_subclass = train['SUBCLASS']

categorical_columns = X.select_dtypes(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# Model Define and Train

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(X_encoded, y_subclass)

# Inference

test_X = test.drop(columns=['ID'])
X_encoded = test_X.copy()
X_encoded[categorical_columns] = ordinal_encoder.transform(test_X[categorical_columns])

predictions = model.predict(X_encoded)

original_labels = le_subclass.inverse_transform(predictions)

# Submisson

submisson = pd.read_csv("./sample_submission.csv")

submisson["SUBCLASS"] = original_labels

submisson.to_csv('./baseline_submission.csv', encoding='UTF-8-sig', index=False)