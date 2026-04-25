import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

st.set_page_config(page_title="ML Preprocessing Tool", layout="wide")
st.title(" Full Preprocessing Stage")

if "data" not in st.session_state:
    st.warning("Please upload your dataset first!")
    st.stop()

df = st.session_state["data"].copy()

tabs = st.tabs([
    "Missing Values", 
    "Encoding & Normalization", 
    "Outliers & Transformation", 
    "Feature Selection & PCA", 
    "Sampling (SMOTE)"
])

# --- 1. MISSING VALUES ---
# --- 1. MISSING VALUES (Updated) ---
with tabs[0]:
    st.subheader("🔍 Missing Values Analysis & Handling")
    
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    
    if not missing_cols.empty:
        st.warning(f"Found {len(missing_cols)} columns with missing values.")
        st.table(missing_cols.rename("Missing Count"))
        
        # --- Option A: Drop Columns ---
        st.markdown("### 🗑️ Option 1: Drop Columns or Rows")
        cols_to_drop = st.multiselect("Select columns to drop entirely", df.columns)
        drop_na_rows = st.checkbox("Drop rows with any missing values")
        
        if st.button("Apply Drop"):
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)
                st.success(f"Dropped columns: {cols_to_drop}")
            if drop_na_rows:
                df.dropna(inplace=True)
                st.success("Dropped rows with missing values")
            st.session_state["data"] = df
            st.rerun()

        st.divider()

        # --- Option B: Imputation ---
        st.markdown("### 🛠️ Option 2: Impute Missing Values")
        cols_missing = st.multiselect("Select columns to impute", missing_cols.index.tolist(), default=missing_cols.index.tolist())
        
        method = st.selectbox("Imputer Type", ["Simple (Mean/Median/Mode)", "KNN", "Iterative"])
        
        # تخصيص الخيارات بناءً على نوع الـ Imputer
        if method == "Simple (Mean/Median/Mode)":
            strategy = st.selectbox("Strategy", ["mean", "median", "most_frequent", "constant"])
            fill_value = None
            if strategy == "constant":
                fill_value = st.text_input("Enter constant value to fill")
        
        if st.button("Apply Imputation"):
            if cols_missing:
                if method == "Simple (Mean/Median/Mode)":
                    # ملاحظة: most_frequent (Mode) بتنفع للـ Categorical والـ Numerical
                    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                elif method == "KNN":
                    imputer = KNNImputer()
                else:
                    imputer = IterativeImputer()
                
                # تنفيذ الـ Imputation
                df[cols_missing] = imputer.fit_transform(df[cols_missing])
                st.session_state["data"] = df
                st.success(f"Imputation applied using {method} strategy!")
                st.rerun()
            else:
                st.error("Please select at least one column.")
    else:
        st.success("No missing values detected! 🎉")

# --- 2. ENCODING & NORMALIZATION ---
with tabs[1]:
    col_enc, col_norm = st.columns(2)
    
    with col_enc:
        st.subheader("🔢 Data Encoding")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            st.info(f"Categorical Columns: {cat_cols}")
            enc_col = st.selectbox("Column to Encode", cat_cols)
            enc_type = st.radio("Type", ["Label Encoder", "One-Hot Encoder"])
            
            if st.button("Run Encoding"):
                if enc_type == "Label Encoder":
                    le = LabelEncoder()
                    df[enc_col] = le.fit_transform(df[enc_col].astype(str))
                else:
                    df = pd.get_dummies(df, columns=[enc_col])
                
                st.session_state["data"] = df
                st.success(f"Encoded {enc_col} successfully!")
                st.rerun()
        else:
            st.success("All columns are numeric! No encoding needed. ✅")

    with col_norm:
        st.subheader("⚖️ Normalization & Scaling")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if num_cols:
            scale_cols = st.multiselect("Columns to Scale", num_cols)
            scale_type = st.radio("Scaler Type", ["Standard Scaler", "MinMax Scaler"])
            
            if st.button("Run Scaling"):
                if scale_cols:
                    # احتفاظ بالداتا قبل الـ Scaling عشان الرسم البياني
                    old_data = df[scale_cols].copy()
                    
                    scaler = StandardScaler() if scale_type == "Standard Scaler" else MinMaxScaler()
                    df[scale_cols] = scaler.fit_transform(df[scale_cols])
                    
                    st.session_state["data"] = df
                    
                    # --- جزء الـ Visualization ---
                    st.write("📈 Visualizing Transformation:")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    
                    # قبل الـ Scaling
                    sns.histplot(old_data, kde=True, ax=ax1)
                    ax1.set_title("Before Scaling")
                    
                    # بعد الـ Scaling
                    sns.histplot(df[scale_cols], kde=True, ax=ax2)
                    ax2.set_title(f"After {scale_type}")
                    
                    st.pyplot(fig)
                    st.success("Scaling Applied!")
                else:
                    st.error("Please select columns to scale.")
# --- 3. OUTLIERS & TRANSFORMATION ---
with tabs[2]:
    st.subheader("🚀 Transformation & Outliers Handling")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- 1. Visual Detection ---
    st.markdown("### 📊 Visual Detection")
    selected_vis_col = st.selectbox("Select column to visualize outliers", num_cols)
    if selected_vis_col:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=df[selected_vis_col], ax=ax[0], color="skyblue")
        ax[0].set_title("Boxplot")
        sns.histplot(df[selected_vis_col], kde=True, ax=ax[1], color="salmon")
        ax[1].set_title("Distribution")
        st.pyplot(fig)

    # --- 2. Outlier Treatment ---
    st.markdown("### 🛠️ Outlier Treatment")
    treatment_method = st.radio("Select Treatment Strategy", ["Detect Only", "Clipping (Manual)", "Winsorization (Percentile)"])
    
    target_outlier_cols = st.multiselect("Select columns to treat", num_cols)

    if treatment_method == "Clipping (Manual)":
        lower_val = st.number_input("Lower Bound (Min)", value=float(df[target_outlier_cols].min().min() if target_outlier_cols else 0))
        upper_val = st.number_input("Upper Bound (Max)", value=float(df[target_outlier_cols].max().max() if target_outlier_cols else 100))
        
        if st.button("Apply Clipping"):
            df[target_outlier_cols] = df[target_outlier_cols].clip(lower=lower_val, upper=upper_val)
            st.session_state["data"] = df
            st.success("Data Clipped Successfully!")
            st.rerun()

    elif treatment_method == "Winsorization (Percentile)":
        p_val = st.slider("Select Percentile to limit (e.g., 0.05 means limits at 5th and 95th)", 0.01, 0.10, 0.05)
        if st.button("Apply Winsorization"):
            for col in target_outlier_cols:
                lower = df[col].quantile(p_val)
                upper = df[col].quantile(1 - p_val)
                df[col] = df[col].clip(lower=lower, upper=upper)
            st.session_state["data"] = df
            st.success(f"Winsorization applied at {p_val*100}%")
            st.rerun()

    st.divider()

    # --- 3. Mathematical Transformation ---
    st.markdown("### 🧬 Mathematical Transformation")
    trans_cols = st.multiselect("Select columns for transformation", num_cols, key="trans_cols")
    trans_type = st.selectbox("Method", ["Log Transformation", "Box-Cox", "Yeo-Johnson"])
    
    if st.button("Apply Transformation"):
        if trans_cols:
            if trans_type == "Log Transformation":
                df[trans_cols] = np.log1p(df[trans_cols])
            elif trans_type == "Box-Cox":
                # Box-Cox requires strictly positive data
                pt = PowerTransformer(method='box-cox')
                df[trans_cols] = pt.fit_transform(df[trans_cols] + 0.00001)
            elif trans_type == "Yeo-Johnson":
                # Yeo-Johnson works with positive and negative data
                pt = PowerTransformer(method='yeo-johnson')
                df[trans_cols] = pt.fit_transform(df[trans_cols])
            
            st.session_state["data"] = df
            st.success(f"{trans_type} Applied!")
            st.rerun()

# --- 4. FEATURE SELECTION (RFE) & PCA ---
with tabs[3]:
    st.subheader("🎯 Professional Feature Selection Hub")
    
    target_col = st.selectbox("Select Target Variable (Y)", df.columns, key="fs_target")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # تجهيز الداتا للحسابات (Encoding مؤقت للـ Categorical)
    X_numeric = X.select_dtypes(include=[np.number])
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
    
    y_encoded = LabelEncoder().fit_transform(y.astype(str)) if y.dtype == 'object' else y

    method_type = st.radio("Selection Method", ["Filter Methods (Statistical)", "Wrapper (RFE)", "Embedded/PCA"])

    # --- 1. FILTER METHODS ---
    if method_type == "Filter Methods (Statistical)":
        stat_choice = st.selectbox("Statistical Test", [
            "Variance Threshold", 
            "Correlation Coefficient", 
            "Chi-Square Test", 
            "Mutual Information"
        ])

        if stat_choice == "Variance Threshold":
            variances = X_numeric.var()
            st.bar_chart(variances)
            threshold = st.number_input("Variance Threshold", value=0.01, format="%.4f")
            selected_features = variances[variances > threshold].index.tolist()

        elif stat_choice == "Correlation Coefficient":
            correlations = X_numeric.corrwith(y_encoded).abs().sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=correlations.values, y=correlations.index, palette="viridis", ax=ax)
            st.pyplot(fig)
            top_k = st.slider("Select Top K Features", 1, len(correlations), min(5, len(correlations)))
            selected_features = correlations.head(top_k).index.tolist()

        elif stat_choice == "Chi-Square Test":
            from sklearn.feature_selection import chi2, SelectKBest
            # Chi-square محتاج بيانات غير سالبة
            X_pos = X_encoded.clip(lower=0) 
            scores, p_values = chi2(X_pos, y_encoded)
            chi_df = pd.Series(scores, index=X_encoded.columns).sort_values(ascending=False)
            st.bar_chart(chi_df)
            top_k = st.slider("Select Top K (Chi2)", 1, len(chi_df), 5)
            selected_features = chi_df.head(top_k).index.tolist()

        elif stat_choice == "Mutual Information":
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            # اختيار الاختبار بناءً على نوع الـ Target
            is_regression = y.dtype != 'object' and y.nunique() > 10
            mi_func = mutual_info_regression if is_regression else mutual_info_classif
            mi_scores = mi_func(X_encoded, y_encoded)
            mi_df = pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)
            st.bar_chart(mi_df)
            top_k = st.slider("Select Top K (Mutual Info)", 1, len(mi_df), 5)
            selected_features = mi_df.head(top_k).index.tolist()

        if st.button("Update Dataset with Filtered Features"):
            st.session_state["data"] = df[selected_features + [target_col]]
            st.success(f"Kept {len(selected_features)} features!")
            st.rerun()

    # --- 2. WRAPPER METHODS (RFE) ---
    elif method_type == "Wrapper (RFE)":
        st.info("RFE recursively removes least important features using a model.")
        n_features = st.slider("Features to Select", 1, len(X_encoded.columns), 5)
        if st.button("Run RFE"):
            model = LogisticRegression(max_iter=1000)
            selector = RFE(model, n_features_to_select=n_features)
            selector = selector.fit(X_encoded, y_encoded)
            selected_features = X_encoded.columns[selector.support_].tolist()
            st.write("✅ Selected:", selected_features)
            st.session_state["data"] = df[selected_features + [target_col]]
            st.rerun()

    # --- 3. PCA ---
    elif method_type == "Embedded/PCA":
        n_comp = st.slider("Number of Components", 1, len(X_numeric.columns), 2)
        if st.button("Apply PCA"):
            pca = PCA(n_components=n_comp)
            pca_res = pca.fit_transform(X_numeric)
            pca_df = pd.DataFrame(pca_res, columns=[f"PC{i+1}" for i in range(n_comp)])
            # Explained Variance Plot
            fig, ax = plt.subplots()
            ax.plot(range(1, n_comp+1), np.cumsum(pca.explained_variance_ratio_), marker='o')
            ax.set_title("Cumulative Explained Variance")
            st.pyplot(fig)
            st.session_state["data"] = pd.concat([pca_df, y.reset_index(drop=True)], axis=1)
            st.success("Dataset replaced with PCA components!")
            st.rerun()
# --- 5. SAMPLING (SMOTE) ---
with tabs[4]:
    st.subheader(" Handling Imbalanced Data")
    sampling_target = st.selectbox("Select Target for Sampling", df.columns, key="smote_target")
    
    class_counts = df[sampling_target].value_counts(normalize=True) * 100
    st.write("Class Distribution Percentage:")
    st.bar_chart(class_counts)
    
    if class_counts.max() > 65:
        st.warning(f"Data is Imbalanced! Majority class '{class_counts.idxmax()}' is {class_counts.max():.2f}%")
    else:
        st.success("Data is well balanced.")

    sampling_type = st.radio("Strategy", ["Oversampling (SMOTE)", "Undersampling"])
    
    if st.button("Run Sampling"):
        X_s = df.drop(sampling_target, axis=1).select_dtypes(include=[np.number])
        y_s = df[sampling_target]
        
        if sampling_type == "Oversampling (SMOTE)":
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X_s, y_s)
        else:
            rus = RandomUnderSampler()
            X_res, y_res = rus.fit_resample(X_s, y_s)
            
        new_df = pd.concat([pd.DataFrame(X_res, columns=X_s.columns), pd.Series(y_res, name=sampling_target)], axis=1)
        st.session_state["data"] = new_df
        st.success(f"Sampling Finished! Original size: {len(df)}, New size: {len(new_df)}")
        st.rerun()