import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer,
    PowerTransformer
)

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE


# =========================
# APP
# =========================
st.title("🚀 Optimized ML Pipeline (Final)")

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.session_state["df"] = df

if "df" not in st.session_state:
    st.stop()

df = st.session_state["df"]
st.dataframe(df.head())

# =========================
# TARGET
# =========================
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() < 20 else None
)

# =========================
# COLUMNS
# =========================
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(exclude=np.number).columns

# =========================
# OPTIONS
# =========================
impute_type = st.selectbox("Imputation", ["Mean", "Median", "KNN"])
scale_type = st.selectbox("Scaling", ["StandardScaler", "MinMaxScaler"])
transform_type = st.selectbox("Transformation", ["None", "Log", "Yeo-Johnson"])

use_smote = st.checkbox("Use SMOTE")
use_fs = st.checkbox("Feature Selection")

k = st.slider("SelectKBest Features", 1, 50, 10)

model_type = st.selectbox(
    "Model",
    ["Logistic Regression", "Random Forest", "SVM"]
)

# =========================
# IMPUTER
# =========================
if impute_type == "KNN":
    imputer = KNNImputer(n_neighbors=5)
elif impute_type == "Median":
    imputer = SimpleImputer(strategy="median")
else:
    imputer = SimpleImputer(strategy="mean")

# =========================
# SCALER
# =========================
scaler = StandardScaler() if scale_type == "StandardScaler" else MinMaxScaler()

# =========================
# TRANSFORMATION
# =========================
transformer = None

if transform_type == "Log":
    transformer = FunctionTransformer(lambda x: np.log1p(np.abs(x)))

elif transform_type == "Yeo-Johnson":
    transformer = PowerTransformer(method="yeo-johnson")

# =========================
# MODEL
# =========================
if model_type == "Random Forest":
    model = RandomForestClassifier()
elif model_type == "SVM":
    model = SVC()
else:
    model = LogisticRegression(max_iter=1000)

# =========================
# PREPROCESSING PIPELINE
# =========================
num_steps = [("imputer", imputer)]

if transformer:
    num_steps.append(("transform", transformer))

num_steps.append(("scaler", scaler))

numeric_pipe = Pipeline(num_steps)

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

# =========================
# TRAIN
# =========================
if st.button(" Train Model"):

    # 1. PREPROCESS
    X_train_p = preprocess.fit_transform(X_train)
    X_test_p = preprocess.transform(X_test)

    # =========================
    # 2. FEATURE SELECTION (BEFORE SMOTE) ✔
    # =========================
    if use_fs:
        k_safe = min(k, X_train_p.shape[1])
        selector = SelectKBest(f_classif, k=k_safe)

        X_train_p = selector.fit_transform(X_train_p, y_train)
        X_test_p = selector.transform(X_test_p)

    # =========================
    # 3. SMOTE (AFTER FEATURE SELECTION) ✔
    # =========================
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train_p, y_train = sm.fit_resample(X_train_p, y_train)

    # =========================
    # 4. MODEL
    # =========================
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)

    acc = model.score(X_test_p, y_test)

    st.success(f" Accuracy: {acc:.4f}")

    # =========================
    # EVALUATION
    # =========================
    st.subheader(" Evaluation")

    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
