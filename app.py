import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import io
from sklearn.ensemble import IsolationForest
from google import genai

st.set_page_config(
    page_title="Audit Analytics: Comprehensive Revenue & AR System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ระบบตรวจสอบและวิเคราะห์ความเสี่ยงวงจรรายได้และลูกหนี้การค้า (Comprehensive AR Audit)")
st.caption("Automated 3-Way Matching, Fraud Detection, Benford's Law, Machine Learning, Cut-off Testing & TFRS 9 ECL")

# ==============================================================================
# 1. แถบควบคุมด้านข้าง (Sidebar)
# ==============================================================================
st.sidebar.header("1. การตั้งค่าระบบและ AI")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="รับได้ฟรีจาก Google AI Studio")

st.sidebar.markdown("---")
st.sidebar.header("2. แหล่งข้อมูล (Data Source)")

@st.cache_data
def generate_audit_template():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        pd.DataFrame(columns=["Customer_ID", "Customer_Name", "Tax_ID", "Address", "Credit_Limit", "Standard_Term"]).to_excel(writer, sheet_name="Customer_Master", index=False)
        pd.DataFrame(columns=["SO_No", "Customer_ID", "Item", "SO_Qty", "Approved_Price", "Approved_Term", "Max_Discount_Pct"]).to_excel(writer, sheet_name="Sales_Order", index=False)
        pd.DataFrame(columns=["DO_No", "SO_No", "Delivered_Qty", "Delivery_Date"]).to_excel(writer, sheet_name="Delivery_Order", index=False)
        pd.DataFrame(columns=["INV_No", "SO_No", "Billed_Qty", "Billed_Price", "Discount_Pct", "INV_Date", "Paid_Amount", "Status", "Cancelled_Date"]).to_excel(writer, sheet_name="Invoice_Ledger", index=False)
    return buffer.getvalue()

st.sidebar.download_button(
    label="📥 ดาวน์โหลดไฟล์ Template Excel",
    data=generate_audit_template(),
    file_name="Audit_Revenue_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="ดาวน์โหลดไฟล์ Excel เปล่าที่มีโครงสร้างหัวตารางครบทั้ง 4 แผ่นงาน"
)

data_mode = st.sidebar.radio(
    "เลือกแหล่งข้อมูล:",
    ["ใช้ข้อมูลทดสอบเชิงลึก (Mock Comprehensive Data)", "อัปโหลดไฟล์ Excel เดี่ยว (Single Multi-Sheet Excel)"]
)
st.sidebar.markdown("---")

@st.cache_data
def generate_mock_datasets():
    customers = pd.DataFrame([
        {"Customer_ID": "CUST-001", "Customer_Name": "บริษัท กิจการรุ่งเรือง จำกัด", "Tax_ID": "0105551000001", "Address": "99/1 ถ.สาทร กทม.", "Credit_Limit": 1000000.0, "Standard_Term": 30},
        {"Customer_ID": "CUST-002", "Customer_Name": "หจก. ทรัพย์ทวี เทรดดิ้ง", "Tax_ID": "0105551000002", "Address": "45/2 ถ.สีลม กทม.", "Credit_Limit": 200000.0, "Standard_Term": 30},
        {"Customer_ID": "CUST-003", "Customer_Name": "บมจ. สยาม ซัพพลาย แอนด์ โลจิสติกส์", "Tax_ID": "0105551000003", "Address": "123 ถ.สุขุมวิท กทม.", "Credit_Limit": 5000000.0, "Standard_Term": 60},
        {"Customer_ID": "CUST-004", "Customer_Name": "บจก. สยาม โกลบอล พาร์ทเนอร์", "Tax_ID": "0105551000003", "Address": "123 ถ.สุขุมวิท กทม.", "Credit_Limit": 100000.0, "Standard_Term": 30},
        {"Customer_ID": "CUST-005", "ร้าน เจริญผล พาณิชย์": "ร้าน เจริญผล พาณิชย์", "Customer_Name": "ร้าน เจริญผล พาณิชย์", "Tax_ID": "0105551000005", "Address": "88 หมู่ 3 นนทบุรี", "Credit_Limit": 300000.0, "Standard_Term": 30},
        {"Customer_ID": "CUST-006", "Customer_Name": "บจก. นอมินี เวิลด์ (Ghost Customer)", "Tax_ID": "0105551000006", "Address": "99/1 ถ.สาทร กทม.", "Credit_Limit": 500000.0, "Standard_Term": 30},
    ])
    so = pd.DataFrame([
        {"SO_No": "SO-101", "Customer_ID": "CUST-001", "Item": "สินค้า Alpha", "SO_Qty": 100, "Approved_Price": 500.0, "Approved_Term": 30, "Max_Discount_Pct": 5.0},
        {"SO_No": "SO-102", "Customer_ID": "CUST-002", "Item": "สินค้า Beta", "SO_Qty": 50, "Approved_Price": 1200.0, "Approved_Term": 30, "Max_Discount_Pct": 0.0},
        {"SO_No": "SO-103", "Customer_ID": "CUST-003", "Item": "สินค้า Gamma", "SO_Qty": 200, "Approved_Price": 350.0, "Approved_Term": 60, "Max_Discount_Pct": 10.0},
        {"SO_No": "SO-104", "Customer_ID": "CUST-004", "Item": "สินค้า Alpha", "SO_Qty": 150, "Approved_Price": 500.0, "Approved_Term": 90, "Max_Discount_Pct": 0.0},
        {"SO_No": "SO-105", "Customer_ID": "CUST-005", "Item": "สินค้า Delta", "SO_Qty": 20, "Approved_Price": 8000.0, "Approved_Term": 30, "Max_Discount_Pct": 5.0},
        {"SO_No": "SO-106", "Customer_ID": "CUST-006", "Item": "สินค้า Alpha", "SO_Qty": 400, "Approved_Price": 500.0, "Approved_Term": 30, "Max_Discount_Pct": 0.0},
    ])
    do = pd.DataFrame([
        {"DO_No": "DO-101", "SO_No": "SO-101", "Delivered_Qty": 100, "Delivery_Date": "2026-06-15"},
        {"DO_No": "DO-102", "SO_No": "SO-102", "Delivered_Qty": 35, "Delivery_Date": "2026-07-10"},
        {"DO_No": "DO-103", "SO_No": "SO-103", "Delivered_Qty": 200, "Delivery_Date": "2026-08-01"},
        {"DO_No": "DO-104", "SO_No": "SO-104", "Delivered_Qty": 150, "Delivery_Date": "2026-08-15"},
        {"DO_No": "DO-105", "SO_No": "SO-105", "Delivered_Qty": 20, "Delivery_Date": "2026-09-25"},
    ])
    inv = pd.DataFrame([
        {"INV_No": "INV-101", "SO_No": "SO-101", "Billed_Qty": 100, "Billed_Price": 500.0, "Discount_Pct": 5.0, "INV_Date": "2026-06-15", "Paid_Amount": 47500.0, "Status": "Normal", "Cancelled_Date": None},
        {"INV_No": "INV-102", "SO_No": "SO-102", "Billed_Qty": 50, "Billed_Price": 1200.0, "Discount_Pct": 0.0, "INV_Date": "2026-07-10", "Paid_Amount": 0.0, "Status": "Normal", "Cancelled_Date": None},
        {"INV_No": "INV-103", "SO_No": "SO-103", "Billed_Qty": 200, "Billed_Price": 380.0, "Discount_Pct": 18.0, "INV_Date": "2026-08-01", "Paid_Amount": 20000.0, "Status": "Normal", "Cancelled_Date": None},
        {"INV_No": "INV-104", "SO_No": "SO-104", "Billed_Qty": 150, "Billed_Price": 500.0, "Discount_Pct": 0.0, "INV_Date": "2026-05-15", "Paid_Amount": 0.0, "Status": "Normal", "Cancelled_Date": None},
        {"INV_No": "INV-106", "SO_No": "SO-106", "Billed_Qty": 400, "Billed_Price": 500.0, "Discount_Pct": 0.0, "INV_Date": "2026-06-30", "Paid_Amount": 0.0, "Status": "Cancelled", "Cancelled_Date": "2026-07-03"},
    ])
    return customers, so, do, inv
    st.sidebar.header("การตั้งค่าระบบและ AI")
api_key = st.sidebar.text_input(...)

st.sidebar.markdown("---")
st.sidebar.header("แหล่งข้อมูล (Data Source)")

# >>> วางโค้ดปุ่มดาวน์โหลด Template ตรงนี้ <<<
template_bytes = generate_audit_template()
st.sidebar.download_button(
    label="📥 ดาวน์โหลดไฟล์ Template Excel",
    data=template_bytes,
    file_name="Audit_Revenue_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


if data_mode == "ใช้ข้อมูลทดสอบเชิงลึก (Mock Comprehensive Data)":
    df_cust, df_so, df_do, df_inv = generate_mock_datasets()
else:
    uploaded_excel = st.sidebar.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx", "xls"])
    if uploaded_excel is not None:
        excel_file = pd.ExcelFile(uploaded_excel)
        sheets = excel_file.sheet_names
        
        def match_sheet(keys):
            for s in sheets:
                if any(k.lower() in s.lower() for k in keys):
                    return sheets.index(s)
            return 0

        # กำหนดตัวเลือก Sheet ทาง Sidebar
        s_cust = st.sidebar.selectbox("Sheet ลูกค้า (Customer Master)", sheets, index=match_sheet(["cust", "ลูกค้า"]))
        s_so = st.sidebar.selectbox("Sheet ใบสั่งขาย (Sales Order / PO)", sheets, index=match_sheet(["so", "sales", "order"]))
        s_do = st.sidebar.selectbox("Sheet ใบส่งของ (Delivery Order)", sheets, index=match_sheet(["do", "delivery", "ส่ง"]))
        s_inv = st.sidebar.selectbox("Sheet ใบแจ้งหนี้ (Invoice / AR)", sheets, index=match_sheet(["invoice", "inv", "ลูกหนี้", "ledger"]))

        # ฟังก์ชันค้นหาแถวที่เป็นหัวตารางจริงอัตโนมัติ (ข้ามแถวหัวเรื่องถ้ามี)
        def smart_read_sheet(file, sheet_name, key_cols):
            temp_df = pd.read_excel(file, sheet_name=sheet_name)
            if any(k in temp_df.columns for k in key_cols):
                return temp_df
            for skip in range(1, 10):
                temp_df = pd.read_excel(file, sheet_name=sheet_name, skiprows=skip)
                if any(k in temp_df.columns for k in key_cols):
                    return temp_df
            return temp_df

        try:
            df_cust = smart_read_sheet(uploaded_excel, s_cust, ["Customer_ID", "Customer_Name"])
            df_so = smart_read_sheet(uploaded_excel, s_so, ["SO_No", "Approved_Price"])
            df_do = smart_read_sheet(uploaded_excel, s_do, ["DO_No", "Delivered_Qty"])
            df_inv = smart_read_sheet(uploaded_excel, s_inv, ["INV_No", "Billed_Qty"])
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ Excel: {str(e)}")
            st.stop()
    else:
        st.info("กรุณาอัปโหลดไฟล์ Excel ที่มีครบทุก Sheet เพื่อเริ่มต้นการวิเคราะห์")
        st.stop()
# ==============================================================================
# 2. เครื่องมือประมวลผลข้อมูล (Core Audit Engine)
# ==============================================================================
# ลบคอลัมน์ Audit_Note จากตารางย่อยเพื่อป้องกันชื่อคอลัมน์ชนกันตอน merge
df_cust_clean = df_cust.drop(columns=["Audit_Note"], errors="ignore")
df_so_clean = df_so.drop(columns=["Audit_Note"], errors="ignore")
df_do_clean = df_do.drop(columns=["Audit_Note"], errors="ignore")
df_inv_clean = df_inv.drop(columns=["Audit_Note"], errors="ignore")

# รวมตารางด้วยคีย์หลัก
m1 = pd.merge(df_so_clean, df_cust_clean, on="Customer_ID", how="left")
m2 = pd.merge(m1, df_do_clean, on="SO_No", how="outer")
audit_master = pd.merge(m2, df_inv_clean, on="SO_No", how="outer")

audit_master["Gross_Amount"] = audit_master["Billed_Qty"] * audit_master["Billed_Price"]
audit_master["Discount_Amount"] = audit_master["Gross_Amount"] * (audit_master["Discount_Pct"].fillna(0) / 100.0)
audit_master["Net_Invoice_Amount"] = audit_master["Gross_Amount"] - audit_master["Discount_Amount"]
audit_master["Outstanding_AR"] = audit_master["Net_Invoice_Amount"] - audit_master["Paid_Amount"].fillna(0)
audit_master["Outstanding_AR"] = audit_master["Outstanding_AR"].apply(lambda x: max(x, 0.0) if pd.notna(x) else 0.0)

audit_master["INV_Date"] = pd.to_datetime(audit_master["INV_Date"])
audit_master["Effective_Term"] = audit_master["Approved_Term"].fillna(audit_master["Standard_Term"]).fillna(30)
audit_master["Due_Date"] = audit_master["INV_Date"] + pd.to_timedelta(audit_master["Effective_Term"], unit="D")

# การตรวจ 3-Way
def eval_3way(r):
    errs = []
    if pd.isna(r["DO_No"]) and pd.notna(r["INV_No"]): errs.append("ไม่มีใบส่งของ (DO ขาด) เสี่ยงเป็นยอดขายทิพย์")
    if pd.notna(r["DO_No"]) and pd.isna(r["INV_No"]): errs.append("ส่งของแล้วแต่ยังไม่ออกบิล (Unbilled Revenue)")
    if pd.notna(r["Delivered_Qty"]) and pd.notna(r["Billed_Qty"]):
        if r["Billed_Qty"] > r["Delivered_Qty"]: errs.append(f"เปิดบิลเกินส่งจริง ({r['Billed_Qty']} > {r['Delivered_Qty']})")
    if pd.notna(r["Approved_Price"]) and pd.notna(r["Billed_Price"]):
        if r["Billed_Price"] > r["Approved_Price"]: errs.append(f"ราคาขายในบิลสูงกว่าที่อนุมัติ ({r['Billed_Price']} > {r['Approved_Price']})")
    return "; ".join(errs) if errs else "ปกติ"

# การตรวจ Credit & Term
def eval_credit(r):
    errs = []
    if pd.notna(r["Approved_Term"]) and pd.notna(r["Standard_Term"]):
        if r["Approved_Term"] > r["Standard_Term"]: errs.append(f"ให้เทอมเกินมาตรฐาน ({r['Approved_Term']} วัน > {r['Standard_Term']} วัน)")
    if pd.notna(r["Outstanding_AR"]) and pd.notna(r["Credit_Limit"]):
        if r["Outstanding_AR"] > r["Credit_Limit"]: errs.append(f"หนี้ค้างเกินวงเงิน ({r['Outstanding_AR']:,.0f} > {r['Credit_Limit']:,.0f})")
    return "; ".join(errs) if errs else "ปกติ"

# การตรวจ Leakage
def eval_leakage(r):
    errs = []
    if pd.notna(r["Discount_Pct"]) and pd.notna(r["Max_Discount_Pct"]):
        if r["Discount_Pct"] > r["Max_Discount_Pct"]: errs.append(f"ส่วนลดเกินเพดาน ({r['Discount_Pct']}% > {r['Max_Discount_Pct']}%)")
    if r.get("Status") == "Cancelled" and pd.notna(r.get("Cancelled_Date")):
        inv_m = r["INV_Date"].month if pd.notna(r["INV_Date"]) else None
        can_d = pd.to_datetime(r["Cancelled_Date"])
        if inv_m and can_d.month != inv_m: errs.append(f"ยกเลิกบิลข้ามงวด (ออกเดือน {inv_m} ยกเลิก {can_d.strftime('%Y-%m-%d')})")
    return "; ".join(errs) if errs else "ปกติ"

audit_master["3Way_Finding"] = audit_master.apply(eval_3way, axis=1)
audit_master["Credit_Finding"] = audit_master.apply(eval_credit, axis=1)
audit_master["Leakage_Finding"] = audit_master.apply(eval_leakage, axis=1)

# ==============================================================================
# 3. ตัวกรองข้อมูลแบบ Interactive (Sidebar Filter)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("3. กรองข้อมูล (Interactive Filters)")
all_custs = ["ทั้งหมด (All)"] + list(audit_master["Customer_Name"].dropna().unique())
sel_cust = st.sidebar.selectbox("เลือกเฉพาะลูกค้า:", all_custs)

valid_dates = audit_master["INV_Date"].dropna()
if not valid_dates.empty:
    min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
    sel_range = st.sidebar.date_input("ช่วงวันที่ออกใบแจ้งหนี้:", [min_d, max_d])
else:
    sel_range = None

# กรอง Master Data ตามเงื่อนไข
filtered_df = audit_master.copy()
if sel_cust != "ทั้งหมด (All)":
    filtered_df = filtered_df[filtered_df["Customer_Name"] == sel_cust]

if sel_range and len(sel_range) == 2:
    start_ts = pd.to_datetime(sel_range[0])
    end_ts = pd.to_datetime(sel_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    
    # กรอง: วันที่อยู่ในช่วง หรือเป็นรายการที่ยังไม่มีวันที่ (Unbilled/ยังไม่ออกบิล)
    is_in_range = (filtered_df["INV_Date"] >= start_ts) & (filtered_df["INV_Date"] <= end_ts)
    is_na_date = filtered_df["INV_Date"].isna()
    filtered_df = filtered_df[is_in_range.fillna(False) | is_na_date]

# ==============================================================================
# 4. ฟังก์ชันส่งออกตารางเป็น Excel (Export Helper)
# ==============================================================================
def to_excel_download(df_to_export):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_to_export.to_excel(writer, index=False, sheet_name='Audit_Exceptions')
    return output.getvalue()

# ==============================================================================
# 5. สรุปภาพรวมแดชบอร์ด (Top Metrics)
# ==============================================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("รายการตรวจทั้งหมด", f"{len(filtered_df)} รายการ")
k2.metric("3-Way Mismatch", f"{(filtered_df['3Way_Finding'] != 'ปกติ').sum()} ประเด็น", delta_color="inverse")
k3.metric("Credit Breach", f"{(filtered_df['Credit_Finding'] != 'ปกติ').sum()} ประเด็น", delta_color="inverse")
k4.metric("Revenue Leakage", f"{(filtered_df['Leakage_Finding'] != 'ปกติ').sum()} ประเด็น", delta_color="inverse")

st.markdown("---")

# ==============================================================================
# 6. แท็บการตรวจสอบ 9 มิติ
# ==============================================================================
tabs = st.tabs([
    "1. 3-Way Matching",
    "2. Credit Controls",
    "3. Ghost Customers",
    "4. Revenue Leakage",
    "5. Cut-off Testing",
    "6. Benford's Law",
    "7. ML Anomaly Detection",
    "8. Aging & ECL (TFRS 9)",
    "9. AR Confirmation",
    "10. AI Audit Assistant"
])

# ----- TAB 1: 3-Way Matching -----
with tabs[0]:
    st.subheader("Automated 3-Way Matching (Sales Order - Delivery - Invoice)")
    ex_3w = filtered_df[filtered_df["3Way_Finding"] != "ปกติ"]
    cols_3w = ["SO_No", "Customer_Name", "DO_No", "INV_No", "SO_Qty", "Delivered_Qty", "Billed_Qty", "Approved_Price", "Billed_Price", "3Way_Finding"]
    if not ex_3w.empty:
        st.dataframe(ex_3w[cols_3w].fillna("-"), use_container_width=True)
        st.download_button("ดาวน์โหลดผลตรวจ 3-Way (.xlsx)", to_excel_download(ex_3w[cols_3w]), "3way_exceptions.xlsx")
    else:
        st.success("ไม่พบข้อผิดปกติในกระบวนการ 3-Way Matching")

# ----- TAB 2: Credit Controls -----
with tabs[1]:
    st.subheader("การอนุมัติวงเงินสินเชื่อและเงื่อนไขระยะเวลาชำระ (Credit Controls)")
    ex_cr = filtered_df[filtered_df["Credit_Finding"] != "ปกติ"]
    cols_cr = ["SO_No", "Customer_Name", "Credit_Limit", "Outstanding_AR", "Standard_Term", "Approved_Term", "Credit_Finding"]
    if not ex_cr.empty:
        st.dataframe(ex_cr[cols_cr].fillna("-"), use_container_width=True)
        st.download_button("ดาวน์โหลดผลตรวจ Credit (.xlsx)", to_excel_download(ex_cr[cols_cr]), "credit_exceptions.xlsx")
    else:
        st.success("วงเงินและเครดิตเทอมเป็นไปตามเกณฑ์มาตรฐาน")

# ----- TAB 3: Ghost Customers -----
with tabs[2]:
    st.subheader("ตรวจสอบลูกหนี้ที่ไม่มีตัวตนจริงและนิติกรรมอำพราง (Ghost Customers)")
    d_tax = df_cust[df_cust.duplicated(subset=["Tax_ID"], keep=False)]
    d_add = df_cust[df_cust.duplicated(subset=["Address"], keep=False)]
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. ลูกค้าที่ใช้เลขประจำตัวผู้เสียภาษี (Tax ID) ซ้ำซ้อน:**")
        if not d_tax.empty:
            st.dataframe(d_tax[["Customer_ID", "Customer_Name", "Tax_ID"]], use_container_width=True)
        else: st.success("ไม่พบ Tax ID ซ้ำ")
    with c2:
        st.markdown("**2. ลูกค้าที่ใช้ที่อยู่จดทะเบียนเดียวกัน (Shared Address):**")
        if not d_add.empty:
            st.dataframe(d_add[["Customer_ID", "Customer_Name", "Address"]], use_container_width=True)
        else: st.success("ไม่พบที่อยู่ซ้ำซ้อน")

# ----- TAB 4: Revenue Leakage -----
with tabs[3]:
    st.subheader("การรั่วไหลของรายได้และการยกเลิกบิลผิดปกติ (Revenue Leakage)")
    ex_lk = filtered_df[filtered_df["Leakage_Finding"] != "ปกติ"]
    cols_lk = ["INV_No", "Customer_Name", "Gross_Amount", "Discount_Pct", "Max_Discount_Pct", "Status", "Cancelled_Date", "Leakage_Finding"]
    if not ex_lk.empty:
        st.dataframe(ex_lk[cols_lk].fillna("-"), use_container_width=True)
        st.download_button("ดาวน์โหลดผลตรวจ Revenue Leakage (.xlsx)", to_excel_download(ex_lk[cols_lk]), "leakage_exceptions.xlsx")
    else:
        st.success("ไม่พบรายการส่วนลดหรือยกเลิกบิลผิดปกติ")

# ----- TAB 5: Cut-off Testing -----
with tabs[4]:
    st.subheader("การตรวจสอบการรับรู้รายได้ข้ามงวดบัญชี (Cut-off Testing)")
    cut_date = st.date_input("กำหนดวันสิ้นงวดบัญชี (Year-End Cut-off Date):", date(2026, 6, 30))
    window = st.slider("ช่วงวันตรวจสอบก่อนและหลังวันสิ้นงวด (วัน):", 3, 30, 7)
    
    s_d = pd.to_datetime(cut_date) - pd.Timedelta(days=window)
    e_d = pd.to_datetime(cut_date) + pd.Timedelta(days=window)
    
    cut_df = filtered_df[(filtered_df["INV_Date"] >= s_d) & (filtered_df["INV_Date"] <= e_d)]
    st.write(f"พบรายการขายในช่วง {s_d.strftime('%Y-%m-%d')} ถึง {e_d.strftime('%Y-%m-%d')} จำนวน {len(cut_df)} รายการ:")
    st.dataframe(cut_df[["INV_No", "Customer_Name", "INV_Date", "Delivery_Date", "Net_Invoice_Amount", "Status"]].fillna("-"), use_container_width=True)

# ----- TAB 6: Benford's Law -----
with tabs[5]:
    st.subheader("การวิเคราะห์ความผิดปกติของตัวเลขด้วยกฎเบนฟอร์ด (Benford's Law)")
    st.caption("เปรียบเทียบการกระจายตัวของเลขหลักแรก (First Digit) ของยอดเงินใน Invoice กับอัตราส่วนมาตรฐานสากล")
    
    valid_inv = filtered_df[filtered_df["Net_Invoice_Amount"] > 0]["Net_Invoice_Amount"].astype(str)
    first_digits = valid_inv.str.extract(r'([1-9])')[0].dropna().astype(int)
    
    if len(first_digits) > 0:
        actual_counts = first_digits.value_counts(normalize=True).sort_index()
        benford_expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}
        
        b_df = pd.DataFrame({
            "Digit": list(range(1, 10)),
            "Actual_Pct": [actual_counts.get(d, 0) * 100 for d in range(1, 10)],
            "Benford_Pct": [benford_expected[d] * 100 for d in range(1, 10)]
        })
        
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(x=b_df["Digit"], y=b_df["Actual_Pct"], name="สัดส่วนที่ตรวจพบจริง (%)", marker_color="royalblue"))
        fig_b.add_trace(go.Scatter(x=b_df["Digit"], y=b_df["Benford_Pct"], mode="lines+markers", name="เกณฑ์มาตรฐาน Benford (%)", line=dict(color="crimson", width=3)))
        fig_b.update_layout(xaxis_title="เลขหลักแรก (1-9)", yaxis_title="สัดส่วนเปอร์เซ็นต์ (%)", barmode="group")
        st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.info("มีข้อมูลตัวเลขยอดเงินไม่เพียงพอสำหรับการทดสอบ Benford's Law")

# ----- TAB 7: Machine Learning Anomaly Detection -----
with tabs[6]:
    st.subheader("การตรวจจับความผิดปกติด้วย Machine Learning (Isolation Forest)")
    st.caption("วิเคราะห์ความผิดปกติแบบพหุมิติ (ยอดเงิน, ปริมาณ, ส่วนลด, เครดิตเทอม) เพื่อหา Outliers")
    
    ml_feats = filtered_df[["Gross_Amount", "Billed_Qty", "Discount_Pct", "Effective_Term"]].fillna(0)
    if len(ml_feats) >= 3:
        iso = IsolationForest(contamination=0.2, random_state=42)
        filtered_df["Anomaly_Score"] = iso.fit_predict(ml_feats)
        anomalies = filtered_df[filtered_df["Anomaly_Score"] == -1]
        
        st.write(f"พบรายการที่มีพฤติกรรมผิดปกติโดย Machine Learning: **{len(anomalies)} รายการ**")
        st.dataframe(anomalies[["INV_No", "Customer_Name", "Gross_Amount", "Billed_Qty", "Discount_Pct", "Effective_Term"]].fillna("-"), use_container_width=True)
    else:
        st.info("จำนวนข้อมูลน้อยเกินไปสำหรับการรันโมเดล Isolation Forest")

    # ----- TAB 8: Aging & TFRS 9 ECL (แก้ไขแล้ว) -----
with tabs[7]:
    st.subheader("วิเคราะห์อายุหนี้และคำนวณค่าเผื่อผลขาดทุนด้านเครดิต (TFRS 9 Provision Matrix)")
    as_of = pd.to_datetime(date.today())
    filtered_df["Overdue_Days"] = (as_of - filtered_df["Due_Date"]).dt.days.fillna(0)
    
    def bucket_ag(d):
        if d <= 0: return "1. ยังไม่ถึงกำหนด"
        elif d <= 30: return "2. เกินกำหนด 1-30 วัน"
        elif d <= 60: return "3. เกินกำหนด 31-60 วัน"
        elif d <= 90: return "4. เกินกำหนด 61-90 วัน"
        else: return "5. เกินกำหนด > 90 วัน (NPL)"
        
    filtered_df["Aging_Group"] = filtered_df["Overdue_Days"].apply(bucket_ag)
    ar_open = filtered_df[filtered_df["Outstanding_AR"] > 0]
    
    loss_rates = {
        "1. ยังไม่ถึงกำหนด": 0.005,
        "2. เกินกำหนด 1-30 วัน": 0.02,
        "3. เกินกำหนด 31-60 วัน": 0.05,
        "4. เกินกำหนด 61-90 วัน": 0.15,
        "5. เกินกำหนด > 90 วัน (NPL)": 0.50
    }

    if not ar_open.empty:
        aging_sum = ar_open.groupby("Aging_Group", as_index=False)["Outstanding_AR"].sum()
        aging_sum["Outstanding_AR"] = pd.to_numeric(aging_sum["Outstanding_AR"], errors="coerce").fillna(0.0)
        
        rates_series = aging_sum["Aging_Group"].map(loss_rates).fillna(0.0)
        aging_sum["Loss_Rate_%"] = rates_series * 100
        aging_sum["ECL_Allowance"] = aging_sum["Outstanding_AR"] * rates_series
        
        cg, ct = st.columns([1, 1])
        with cg:
            fig_ar = px.bar(aging_sum, x="Aging_Group", y="Outstanding_AR", title="ยอดลูกหนี้ตามชั้นอายุ", color="Aging_Group")
            st.plotly_chart(fig_ar, use_container_width=True)
        with ct:
            st.dataframe(aging_sum.style.format({"Outstanding_AR": "{:,.2f}", "Loss_Rate_%": "{:.2f}%", "ECL_Allowance": "{:,.2f}"}), use_container_width=True)
            st.metric("ประมาณการค่าเผื่อ ECL รวมสุทธิ", f"{aging_sum['ECL_Allowance'].sum():,.2f} บาท")
    else:
        st.success("ลูกหนี้รายนี้ไม่มีรายการหนี้ค้างชำระ (Outstanding AR = 0.00 บาท) จึงไม่ต้องตั้งค่าเผื่อผลขาดทุนด้านเครดิต (ECL)")

# ----- TAB 9: AR Confirmation -----
with tabs[8]:
    st.subheader("ระบบสร้างและติดตามหนังสือยืนยันยอดลูกหนี้ (AR Confirmation)")
    st.caption("สุ่มตัวอย่างลูกหนี้ที่มียอดค้างชำระเพื่อจัดทำหนังสือยืนยันยอด (Circularization)")
    
    top_ar = filtered_df[filtered_df["Outstanding_AR"] > 0].sort_values(by="Outstanding_AR", ascending=False)
    if not top_ar.empty:
        target_cust = st.selectbox("เลือกลูกหนี้เพื่อออกหนังสือยืนยันยอด:", top_ar["Customer_Name"].unique())
        cust_row = top_ar[top_ar["Customer_Name"] == target_cust].iloc[0]
        
        conf_letter = f"""
**หนังสือขอคำยืนยันยอดลูกหนี้การค้า (Accounts Receivable Confirmation Request)**
วันที่: {date.today().strftime('%d/%m/%Y')}
เรียน: ฝ่ายบัญชีและการเงิน {cust_row['Customer_Name']}
ที่อยู่: {cust_row.get('Address', '-')}

ในการตรวจสอบบัญชีประจำปีของเรา ขอความกรุณาท่านโปรดยืนยันความถูกต้องของยอดหนี้คงค้าง ณ วันที่ {date.today().strftime('%d/%m/%Y')} 
ซึ่งตามสมุดบัญชีของเรา ปรากฏยอดหนี้ที่ท่านค้างชำระเป็นจำนวนเงินทั้งสิ้น **{cust_row['Outstanding_AR']:,.2f} บาท**

หากยอดดังกล่าว **ถูกต้อง** หรือ **มีข้อทักท้วงประการใด** โปรดลงลายมือชื่อและส่งกลับมายังผู้สอบบัญชีโดยตรง
"""
        st.markdown(conf_letter)
        st.selectbox("บันทึกสถานะการติดตามผล (Confirmation Status Tracker):", 
                     ["1. รอจัดส่งเอกสาร", "2. ส่งเอกสารแล้ว รอการตอบกลับ", "3. ตอบกลับแล้ว ยอดถูกต้องตรงกัน", "4. ตอบกลับแล้ว พบผลต่าง (Disputed)", "5. ไม่ตอบกลับ (Second Request)"])
    else:
        st.info("ไม่พบลูกหนี้ที่มียอดคงค้างสำหรับทำหนังสือยืนยันยอด")

# ----- TAB 10: AI Audit Assistant -----
with tabs[9]:
    st.subheader("AI Internal Audit Assistant (Gemini 2.5)")
    
    issues_list = []
    for idx, r in filtered_df.iterrows():
        f = []
        if r["3Way_Finding"] != "ปกติ": f.append(f"[3-Way] {r['3Way_Finding']}")
        if r["Credit_Finding"] != "ปกติ": f.append(f"[Credit] {r['Credit_Finding']}")
        if r["Leakage_Finding"] != "ปกติ": f.append(f"[Leakage] {r['Leakage_Finding']}")
        if f:
            issues_list.append({"SO_No": r.get("SO_No", "-"), "Customer": r.get("Customer_Name", "-"), "Issues": " | ".join(f)})
            
    df_iss = pd.DataFrame(issues_list)
    
    if st.button("สั่งให้ AI ร่างรายงานการตรวจสอบภายในฉบับเต็ม", type="primary"):
        if not api_key:
            st.error("กรุณาระบุ Gemini API Key ที่แถบด้านซ้ายก่อนใช้งาน")
        elif df_iss.empty:
            st.success("ไม่พบประเด็นข้อผิดพลาด ข้อมูลเป็นไปตามการควบคุมภายใน")
        else:
            with st.spinner("AI กำลังวิเคราะห์ผลกระทบทางบัญชีและร่างรายงานตามมาตรฐาน 5Cs..."):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
คุณเป็นหัวหน้าฝ่ายตรวจสอบภายใน (Chief Audit Executive) ให้วิเคราะห์ข้อบกพร่องเหล่านี้:
{df_iss.to_markdown(index=False)}

กรุณาร่างรายงานการตรวจสอบภายใน (5Cs Framework):
1. Executive Summary
2. Detailed Findings (Criteria, Condition, Cause, Consequence, Corrective Action)
3. Recommendations สำหรับฝ่ายบริหาร
"""
                    res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.session_state["audit_report_text"] = res.text
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: {str(e)}")

    if "audit_report_text" in st.session_state:
        st.markdown(st.session_state["audit_report_text"])
        st.download_button("ดาวน์โหลดรายงาน (.md)", st.session_state["audit_report_text"], "audit_report.md")

    st.markdown("---")
    st.markdown("##### ซักถามข้อสงสัยเพิ่มเติมกับ AI (Interactive Audit Chat)")
    user_q = st.text_input("พิมพ์คำถามที่ต้องการปรึกษา AI เพิ่มเติม:")
    if st.button("ส่งคำถาม"):
        if not api_key:
            st.error("กรุณาระบุ Gemini API Key ก่อน")
        elif user_q:
            with st.spinner("กำลังคิดคำตอบ..."):
                try:
                    client = genai.Client(api_key=api_key)
                    chat_res = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"ในฐานะผู้เชี่ยวชาญด้านการตรวจสอบบัญชีและควบคุมภายใน จงตอบคำถามนี้โดยอิงจากข้อตรวจพบในวงจรรายได้: {user_q}"
                    )
                    st.info(chat_res.text)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
