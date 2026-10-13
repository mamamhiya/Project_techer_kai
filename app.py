from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ARTIFACT_PATH = Path(__file__).resolve().parent / "best_failure_model.joblib"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️")
st.title("ทำนายการขัดข้องของเครื่องจักร")
st.write(
    "กรอกข้อมูลการทำงานของเครื่องเพื่อประเมินโอกาสเกิด failure "
    "(1 = ขัดข้อง, 0 = ไม่ขัดข้อง)"
)

if not ARTIFACT_PATH.exists():
    st.error("ยังไม่พบไฟล์ best_failure_model.joblib")
    st.info("รันเซลล์บันทึกโมเดลท้าย project.ipynb ก่อน แล้วเปิดแอปใหม่")
    st.stop()

model = load_model(str(ARTIFACT_PATH))
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
            "Maintenance days ago", min_value=0.0, value=60.0, step=1.0
        )
        machine_type = st.selectbox("Machine type", ["A", "B", "C"])

    submitted = st.form_submit_button("ทำนาย")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "temperature": temperature,
                "vibration": vibration,
                "pressure": pressure,
                "operating_hours": operating_hours,
                "maintenance_days_ago": maintenance_days_ago,
                "machine_type": machine_type,
            }
        ]
    )
    prediction = int(model.predict(input_data)[0])
    probability = model.predict_proba(input_data)[0]
    failure_probability = float(probability[list(model.classes_).index(1)])

    if prediction == 1:
        st.error("ผลทำนาย: มีแนวโน้มขัดข้อง (failure = 1)")
    else:
        st.success("ผลทำนาย: ไม่พบแนวโน้มขัดข้อง (failure = 0)")
    st.metric("ความน่าจะเป็นที่เครื่องขัดข้อง", f"{failure_probability:.1%}")
    st.caption("ผลทำนายเป็นการประเมินจากโมเดล ไม่ใช่การยืนยันสภาพเครื่องจักร")
