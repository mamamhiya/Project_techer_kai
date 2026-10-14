from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score,
                             adjusted_rand_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "best_failure_model.joblib"
DATA_PATH = BASE_DIR / "C08_machine_failure_classification.xlsx - Raw_Data.csv"
FEATURES = ["temperature", "vibration", "pressure", "operating_hours",
            "maintenance_days_ago", "machine_type"]


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path)
    data = data.dropna(subset=["failure"]).drop_duplicates().copy()
    data["failure"] = data["failure"].astype(int)
    # Keep missing categories as np.nan; pandas StringDtype's pd.NA is not
    # handled consistently by sklearn imputers on every supported version.
    data["machine_type"] = (
        data["machine_type"].astype(object).str.strip().str.upper()
        .replace("", np.nan)
    )
    return data


@st.cache_data
def evaluate_models(data: pd.DataFrame):
    features = [column for column in FEATURES if column in data.columns]
    X, y = data[features], data["failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()
    numeric_scaled = Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())])
    numeric_plain = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    setups = {
        "Logistic Regression": (numeric_scaled, LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42), {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["liblinear", "lbfgs"],
            }),
        "Random Forest": (numeric_plain, RandomForestClassifier(
            random_state=42, n_jobs=1), {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": ["balanced", "balanced_subsample"],
            }),
    }
    rows, matrices = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, (numeric_pipe, estimator, param_grid) in setups.items():
        prep = ColumnTransformer([
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ])
        pipeline = Pipeline([("preprocess", prep), ("model", estimator)])
        search = GridSearchCV(
            pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1,
            refit=True, error_score="raise"
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (failure=1)": precision_score(y_test, pred, zero_division=0),
            "Recall (failure=1)": recall_score(y_test, pred, zero_division=0),
            "F1 (failure=1)": f1_score(y_test, pred, zero_division=0),
            "CV F1 mean": search.best_score_,
            "CV F1 std": search.cv_results_[search.best_index_]["std_test_score"],
            "Best parameters": search.best_params_,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
        matrices[name] = pd.DataFrame(
            [[tn, fp], [fn, tp]], index=["จริง: ปกติ", "จริง: ขัดข้อง"],
            columns=["ทำนาย: ปกติ", "ทำนาย: ขัดข้อง"])
    return pd.DataFrame(rows).set_index("Model").sort_values(
        "CV F1 mean", ascending=False), matrices


st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️", layout="wide")
st.title("ทำนายและวิเคราะห์การขัดข้องของเครื่องจักร")
st.caption("ผลทำนายเป็นการประเมินจากโมเดล ไม่ใช่การยืนยันสภาพเครื่องจักร")

if not ARTIFACT_PATH.exists():
    st.error("ยังไม่พบไฟล์ best_failure_model.joblib")
    st.info("รันเซลล์บันทึกโมเดลท้าย project.ipynb ก่อน แล้วเปิดแอปใหม่")
    st.stop()

model = load_model(str(ARTIFACT_PATH))
tabs = st.tabs(["ทำนาย", "ข้อมูลและ EDA", "ประเมินโมเดล", "Clustering"])

with tabs[0]:
    estimator_name = type(model.named_steps["model"]).__name__
    st.caption(f"โมเดลที่ใช้งาน: {estimator_name}")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input("Temperature", value=72.0, step=0.1)
            vibration = st.number_input("Vibration", value=4.0, step=0.1)
            pressure = st.number_input("Pressure", value=32.0, step=0.1)
        with col2:
            operating_hours = st.number_input("Operating hours", value=900.0, step=1.0)
            maintenance_days_ago = st.number_input(
                "Maintenance days ago", min_value=0.0, value=60.0, step=1.0)
            machine_type = st.selectbox("Machine type", ["A", "B", "C"])
        submitted = st.form_submit_button("ทำนาย")
    if submitted:
        input_data = pd.DataFrame([{
            "temperature": temperature, "vibration": vibration,
            "pressure": pressure, "operating_hours": operating_hours,
            "maintenance_days_ago": maintenance_days_ago,
            "machine_type": machine_type,
        }])
        prediction = int(model.predict(input_data)[0])
        probability = model.predict_proba(input_data)[0]
        failure_probability = float(probability[list(model.classes_).index(1)])
        if prediction == 1:
            st.error("ผลทำนาย: มีแนวโน้มขัดข้อง (failure = 1)")
        else:
            st.success("ผลทำนาย: ไม่พบแนวโน้มขัดข้อง (failure = 0)")
        st.metric("ความน่าจะเป็นที่เครื่องขัดข้อง", f"{failure_probability:.1%}")

if not DATA_PATH.exists():
    for tab in tabs[1:]:
        with tab:
            st.warning(f"ไม่พบไฟล์ข้อมูลสำหรับการวิเคราะห์: {DATA_PATH.name}")
    st.stop()

data = load_data(str(DATA_PATH))
with tabs[1]:
    st.subheader("ภาพรวมข้อมูล")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนแถวหลังทำความสะอาด", f"{len(data):,}")
    c2.metric("จำนวนคอลัมน์", len(data.columns))
    c3.metric("แถวซ้ำที่ตัดออก", f"{pd.read_csv(DATA_PATH).duplicated().sum():,}")
    c4.metric("ข้อมูลขัดข้อง", f"{int(data['failure'].sum()):,} ({data['failure'].mean():.1%})")
    st.markdown("**สัดส่วนสถานะ Failure**")
    counts = data["failure"].value_counts().reindex([0, 1], fill_value=0)
    st.bar_chart(counts.rename(index={0: "ปกติ (0)", 1: "ขัดข้อง (1)"}))
    st.markdown("**อัตราขัดข้องตามประเภทเครื่อง**")
    type_summary = data.groupby("machine_type", dropna=False)["failure"].agg(
        ["count", "mean"]).rename(columns={"count": "จำนวนตัวอย่าง", "mean": "อัตราขัดข้อง"})
    st.dataframe(type_summary.style.format({"อัตราขัดข้อง": "{:.1%}"}), use_container_width=True)
    numeric = [c for c in data.select_dtypes(include="number").columns if c != "failure"]
    st.markdown("**ตัวแปรการทำงานแยกตาม Failure**")
    selected = st.selectbox("เลือกตัวแปร", numeric)
    st.bar_chart(data.groupby("failure")[selected].median().rename(
        index={0: "ปกติ (0)", 1: "ขัดข้อง (1)"}))
    with st.expander("สถิติเบื้องต้นและข้อมูลที่ต้องตรวจสอบ"):
        st.markdown("สถิติเชิงพรรณนา (ค่าผิดปกติตาม IQR เป็นจุดให้ตรวจสอบ ไม่ได้หมายความว่าต้องลบทิ้ง)")
        st.dataframe(data.describe(include="all").T, use_container_width=True)
        missing = data.isna().sum().rename("จำนวนค่าว่าง").to_frame()
        st.dataframe(missing, use_container_width=True)
        outliers = []
        for col in numeric:
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers.append({"ตัวแปร": col, "จำนวนตามเกณฑ์ IQR": int(
                ((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).sum())})
        st.dataframe(pd.DataFrame(outliers).set_index("ตัวแปร"), use_container_width=True)

with tabs[2]:
    st.subheader("เปรียบเทียบ Logistic Regression และ Random Forest")
    st.caption("จูนพารามิเตอร์ด้วย 5-fold Stratified CV บน train 80%; ชุด test 20% ใช้ประเมินครั้งสุดท้าย")
    metrics, matrices = evaluate_models(data)
    display_metrics = metrics[["Accuracy", "Precision (failure=1)",
                               "Recall (failure=1)", "F1 (failure=1)",
                               "CV F1 mean", "CV F1 std"]]
    st.dataframe(display_metrics.style.format("{:.3f}"), use_container_width=True)
    best_name = metrics.index[0]
    best = metrics.iloc[0]
    st.info(f"CV F1 สูงสุด: {best_name} — Test F1 {best['F1 (failure=1)']:.3f}, "
            f"Precision {best['Precision (failure=1)']:.3f}, "
            f"Recall {best['Recall (failure=1)']:.3f}, F1 {best['F1 (failure=1)']:.3f}")
    with st.expander("พารามิเตอร์ที่เลือกจาก Cross-validation"):
        st.json(metrics["Best parameters"].to_dict())
    for name, matrix in matrices.items():
        st.markdown(f"**Confusion Matrix: {name}** (แถว=ค่าจริง, คอลัมน์=ค่าทำนาย)")
        st.dataframe(matrix, use_container_width=True)
        row = metrics.loc[name]
        st.caption(f"TN={row['TN']:.0f}, FP={row['FP']:.0f}, FN={row['FN']:.0f}, TP={row['TP']:.0f}")
    st.caption("คลาสขัดข้องมีจำนวนน้อย ควรพิจารณา Recall/F1 และต้นทุนของ FN กับ FP ร่วมกับ Accuracy")

with tabs[3]:
    st.subheader("การจัดกลุ่มเครื่องจักร")
    st.write("K-Means และ Agglomerative จัดกลุ่มจาก features โดยไม่ใช้ failure ในการสร้างกลุ่ม; ใช้ failure เพียงเปรียบเทียบภายหลัง")
    features = [column for column in FEATURES if column in data.columns]
    X_cluster = data[features]
    numeric = X_cluster.select_dtypes(include="number").columns.tolist()
    categorical = X_cluster.select_dtypes(exclude="number").columns.tolist()
    cluster_preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    matrix = cluster_preprocess.fit_transform(X_cluster)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    labels_by_method = {
        "K-Means": KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(matrix),
        "Agglomerative": AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(matrix),
    }
    cluster_rows = []
    for name, labels in labels_by_method.items():
        cluster_rows.append({
            "วิธี": name,
            "Silhouette": silhouette_score(matrix, labels),
            "ARI เทียบกับ failure": adjusted_rand_score(data["failure"], labels),
        })
    st.markdown("**คุณภาพของกลุ่ม**")
    st.dataframe(pd.DataFrame(cluster_rows).set_index("วิธี").style.format("{:.3f}"),
                 use_container_width=True)
    for name, labels in labels_by_method.items():
        st.markdown(f"**จำนวน Failure ในแต่ละกลุ่ม: {name}**")
        table = pd.crosstab(pd.Series(labels, name="Cluster"),
                            data["failure"].rename("Failure"), margins=True)
        st.dataframe(table, use_container_width=True)
    st.caption("Silhouette วัดความแยกจากกันของกลุ่ม; ARI วัดความสอดคล้องกับ failure ภายหลัง เลข cluster ไม่ได้แทนคลาส failure โดยตรง")
