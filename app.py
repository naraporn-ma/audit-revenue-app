import streamlit as st
import pandas as pd
from google import genai
from google.genai import types

st.set_page_config(page_title="Audit: 3-Way Matching", layout="wide")
st.title("ระบบตรวจสอบ 3-Way Matching และร่างรายงานด้วย AI")
st.caption("วงจรรายได้และลูกหนี้: Sales Order vs Delivery vs Invoice")

# 1. กล่องกรอก API Key ที่แถบด้านซ้าย
st.sidebar.header("การตั้งค่า")
api_key = st.sidebar.text_input("ใส่ Google Gemini API Key", type="password", help="รับได้ฟรีจาก Google AI Studio")

# 2. ข้อมูลจำลองสำหรับทดสอบ
@st.cache_data
def get_sample_data():
    so_data = pd.DataFrame([
        {"SO_No": "SO-001", "Customer": "บริษัท เอ บิซ จำกัด", "SO_Qty": 100, "SO_Price": 500.0},
        {"SO_No": "SO-002", "Customer": "หจก. บี เทรดดิ้ง", "SO_Qty": 50, "SO_Price": 1200.0},
        {"SO_No": "SO-003", "Customer": "บมจ. ซี ซัพพลาย", "SO_Qty": 200, "SO_Price": 350.0},
        {"SO_No": "SO-004", "Customer": "ร้าน ดี พาณิชย์", "SO_Qty": 80, "SO_Price": 500.0},
        {"SO_No": "SO-005", "Customer": "บจก. อี คอร์ป", "SO_Qty": 30, "SO_Price": 2500.0},
    ])
    do_data = pd.DataFrame([
        {"DO_No": "DO-001", "SO_No": "SO-001", "Delivered_Qty": 100},
        {"DO_No": "DO-002", "SO_No": "SO-002", "Delivered_Qty": 40}, # ส่งขาด 10
        {"DO_No": "DO-003", "SO_No": "SO-003", "Delivered_Qty": 200},
        {"DO_No": "DO-005", "SO_No": "SO-005", "Delivered_Qty": 30},
    ])
    inv_data = pd.DataFrame([
        {"INV_No": "INV-001", "SO_No": "SO-001", "Billed_Qty": 100, "Billed_Price": 500.0}, # ตรงสมบูรณ์
        {"INV_No": "INV-002", "SO_No": "SO-002", "Billed_Qty": 50, "Billed_Price": 1200.0}, # บิลเกินยอดส่ง
        {"INV_No": "INV-003", "SO_No": "SO-003", "Billed_Qty": 200, "Billed_Price": 380.0}, # ราคาบิลเกิน SO
        {"INV_No": "INV-004", "SO_No": "SO-004", "Billed_Qty": 80, "Billed_Price": 500.0}, # ไม่มีใบส่งของ
    ])
    return so_data, do_data, inv_data

df_so, df_do, df_inv = get_sample_data()

# 3. ตรวจสอบการกระทบยอด
merged_df = pd.merge(df_so, df_do, on="SO_No", how="outer")
matched_df = pd.merge(merged_df, df_inv, on="SO_No", how="outer")

def evaluate_matching(row):
    reasons = []
    if pd.isna(row["DO_No"]) and pd.notna(row["INV_No"]):
        reasons.append("ไม่มีหลักฐานการส่งของ (DO ขาด) แต่เปิดบิลแล้ว")
    if pd.notna(row["DO_No"]) and pd.isna(row["INV_No"]):
        reasons.append("ส่งของแล้วแต่ยังไม่ได้ออกบิล")
    if pd.notna(row["Delivered_Qty"]) and pd.notna(row["Billed_Qty"]):
        if row["Billed_Qty"] > row["Delivered_Qty"]:
            reasons.append(f"เรียกเก็บเงินเกินยอดส่งจริง ({row['Billed_Qty']} > {row['Delivered_Qty']})")
    if pd.notna(row["SO_Price"]) and pd.notna(row["Billed_Price"]):
        if row["SO_Price"] != row["Billed_Price"]:
            reasons.append(f"ราคาไม่ตรงกับ SO (บิล={row['Billed_Price']}, SO={row['SO_Price']})")
            
    if not reasons:
        return "ถูกต้อง", "ปกติ"
    return "พบข้อผิดปกติ", "; ".join(reasons)

matched_df[["Audit_Status", "Audit_Finding"]] = matched_df.apply(evaluate_matching, axis=1, result_type="expand")

# 4. แสดงผลแดชบอร์ด
exceptions_df = matched_df[matched_df["Audit_Status"] != "ถูกต้อง"]

col1, col2 = st.columns(2)
col1.metric("รายการตรวจทั้งหมด", f"{len(matched_df)} รายการ")
col2.metric("พบข้อผิดปกติ (Exceptions)", f"{len(exceptions_df)} รายการ")

st.subheader("ตารางรายการที่พบข้อผิดปกติ")
display_cols = ["SO_No", "Customer", "DO_No", "INV_No", "SO_Qty", "Delivered_Qty", "Billed_Qty", "SO_Price", "Billed_Price", "Audit_Finding"]
st.dataframe(exceptions_df[display_cols].fillna("-"), use_container_width=True)

# 5. ฟังก์ชันเรียก AI ร่างรายงาน
st.markdown("---")
st.subheader("ให้ AI ช่วยเขียนรายงานข้อตรวจพบ (Audit Findings)")

if st.button("กดให้ AI ร่างรายงาน", type="primary"):
    if not api_key:
        st.error("กรุณานำ Gemini API Key มาวางในช่องด้านซ้ายมือเสียก่อน")
    else:
        with st.spinner("AI กำลังวิเคราะห์และร่างรายงานข้อตรวจพบ..."):
            try:
                client = genai.Client(api_key=api_key)
                summary_table = exceptions_df[display_cols].fillna("-").to_markdown(index=False)
                prompt = f"""
คุณเป็นผู้ตรวจสอบบัญชี ให้วิเคราะห์ข้อผิดพลาดจากการกระทบยอดนี้ และสรุปรายงาน:
{summary_table}

กรุณาร่างรายงานโดยมี:
1. บทสรุปภาพรวมความเสี่ยง
2. รายละเอียดข้อตรวจพบ (ระบุเคส, สภาพการณ์, ผลกระทบต่อความเสี่ยง เช่น เสี่ยงต่อการรับรู้รายได้เกินจริง)
3. ข้อเสนอแนะในการปรับปรุงการควบคุมภายใน
"""
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                )
                st.markdown(response.text)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
