import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, date
from google import genai

st.set_page_config(
    page_title="Audit Analytics: Comprehensive Revenue & AR System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ระบบตรวจสอบและวิเคราะห์ความเสี่ยงวงจรรายได้และลูกหนี้การค้า")
st.caption("Automated 3-Way Matching, Credit Control, Ghost Customers, Revenue Leakage & TFRS 9 ECL")

# ==============================================================================
# 1. แถบควบคุมด้านข้าง (Sidebar)
# ==============================================================================
st.sidebar.header("การตั้งค่าระบบและ API")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="รับได้ฟรีจาก Google AI Studio")

st.sidebar.markdown("---")
st.sidebar.header("แหล่งข้อมูล (Data Source)")
data_mode = st.sidebar.radio("เลือกแหล่งข้อมูล:", ["ใช้ข้อมูลทดสอบเชิงลึก (Mock Comprehensive Data)", "อัปโหลดไฟล์จริงจากระบบ ERP"])

# ฟังก์ชันสร้างชุดข้อมูลทดสอบที่จำลองสถานการณ์จริง
@st.cache_data
def generate_mock_datasets():
    # 1. ลูกค้าและวงเงิน (Customer Master)
    customers = pd.DataFrame([
        {"Customer_ID": "CUST-001", "Customer_Name": "บริษัท กิจการรุ่งเรือง จำกัด", "Tax_ID": "0105551000001", "Address": "99/1 ถ.สาทร กทม.", "Credit_Limit": 1000000.0, "Standard_Term": 30},
        {"Customer_ID": "CUST-002", "Customer_Name": "หจก. ทรัพย์ทวี เทรดดิ้ง", "Tax_ID": "0105551000002", "Address": "45/2 ถ.สีลม กทม.", "Credit_Limit": 200000.0, "Standard_Term": 30},
        {"Customer_ID": "CUST-003", "Customer_Name": "บมจ. สยาม ซัพพลาย แอนด์ โลจิสติกส์", "Tax_ID": "0105551000003", "Address": "123 ถ.สุขุมวิท กทม.", "Credit_Limit": 5000000.0, "Standard_Term": 60},
        {"Customer_ID": "CUST-004", "Customer_Name": "บจก. สยาม โกลบอล พาร์ทเนอร์ (กลุ่มเดียวกับ CUST-003)", "Tax_ID": "0105551000003", "Address": "123 ถ.สุขุมวิท กทม.", "Credit_Limit": 100000.0, "Standard_Term": 30}, # Tax ID ซ้ำ / ที่อยู่ซ้ำ
        {"Customer_ID": "CUST-005", "Customer_Name": "ร้าน เจริญผล พาณิชย์", "Tax_ID": "0105551000005", "Address": "88 หมู่ 3 นนทบุรี", "Credit_Limit": 300000.0, "Standard_Term": 30},
        {"Customer_ID": "CUST-006", "Customer_Name": "บจก. นอมินี เวิลด์ (Ghost Customer)", "Tax_ID": "0105551000006", "Address": "99/1 ถ.สาทร กทม.", "Credit_Limit": 500000.0, "Standard_Term": 30}, # ที่อยู่ตรงกับ CUST-001
    ])

    # 2. ใบสั่งซื้อ/สั่งขาย (PO / SO)
    so = pd.DataFrame([
        {"SO_No": "SO-101", "Customer_ID": "CUST-001", "Item": "สินค้า Alpha", "SO_Qty": 100, "Approved_Price": 500.0, "Approved_Term": 30, "Max_Discount_Pct": 5.0},
        {"SO_No": "SO-102", "Customer_ID": "CUST-002", "Item": "สินค้า Beta", "SO_Qty": 50, "Approved_Price": 1200.0, "Approved_Term": 30, "Max_Discount_Pct": 0.0},
        {"SO_No": "SO-103", "Customer_ID": "CUST-003", "Item": "สินค้า Gamma", "SO_Qty": 200, "Approved_Price": 350.0, "Approved_Term": 60, "Max_Discount_Pct": 10.0},
        {"SO_No": "SO-104", "Customer_ID": "CUST-004", "Item": "สินค้า Alpha", "SO_Qty": 150, "Approved_Price": 500.0, "Approved_Term": 90, "Max_Discount_Pct": 0.0}, # ให้เทอม 90 วันทั้งที่ Standard 30
        {"SO_No": "SO-105", "Customer_ID": "CUST-005", "Item": "สินค้า Delta", "SO_Qty": 20, "Approved_Price": 8000.0, "Approved_Term": 30, "Max_Discount_Pct": 5.0},
        {"SO_No": "SO-106", "Customer_ID": "CUST-006", "Item": "สินค้า Alpha", "SO_Qty": 400, "Approved_Price": 500.0, "Approved_Term": 30, "Max_Discount_Pct": 0.0},
    ])

    # 3. ใบส่งของ (Delivery Order)
    do = pd.DataFrame([
        {"DO_No": "DO-101", "SO_No": "SO-101", "Delivered_Qty": 100, "Delivery_Date": "2026-06-15"},
        {"DO_No": "DO-102", "SO_No": "SO-102", "Delivered_Qty": 35, "Delivery_Date": "2026-07-10"}, # ส่งขาด 15 ชิ้น
        {"DO_No": "DO-103", "SO_No": "SO-103", "Delivered_Qty": 200, "Delivery_Date": "2026-08-01"},
        {"DO_No": "DO-104", "SO_No": "SO-104", "Delivered_Qty": 150, "Delivery_Date": "2026-08-15"},
        # SO-106 ไม่มี DO (เสี่ยงเป็นบิลลวง)
    ])

    # 4. ใบแจ้งหนี้ (Invoice)
    inv = pd.DataFrame([
        {"INV_No": "INV-101", "SO_No": "SO-101", "Billed_Qty": 100, "Billed_Price": 500.0, "Discount_Pct": 5.0, "INV_Date": "2026-06-15", "Paid_Amount": 47500.0, "Status": "Normal", "Cancelled_Date": None},
        {"INV_No": "INV-102", "SO_No": "SO-102", "Billed_Qty": 50, "Billed_Price": 1200.0, "Discount_Pct": 0.0, "INV_Date": "2026-07-10", "Paid_Amount": 0.0, "Status": "Normal", "Cancelled_Date": None}, # บิลเกินยอดส่ง
        {"INV_No": "INV-103", "SO_No": "SO-103", "Billed_Qty": 200, "Billed_Price": 380.0, "Discount_Pct": 18.0, "INV_Date": "2026-08-01", "Paid_Amount": 20000.0, "Status": "Normal", "Cancelled_Date": None}, # ราคาเกิน + ส่วนลดเกิน
        {"INV_No": "INV-104", "SO_No": "SO-104", "Billed_Qty": 150, "Billed_Price": 500.0, "Discount_Pct": 0.0, "INV_Date": "2026-08-15", "Paid_Amount": 0.0, "Status": "Normal", "Cancelled_Date": None},
        {"INV_No": "INV-106", "SO_No": "SO-106", "Billed_Qty": 400, "Billed_Price": 500.0, "Discount_Pct": 0.0, "INV_Date": "2026-06-30", "Paid_Amount": 0.0, "Status": "Cancelled", "Cancelled_Date": "2026-07-03"}, # บิลลอย สิ้นงวดแล้วกดยกเลิก
    ])
    return customers, so, do, inv

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

if data_mode == "ใช้ข้อมูลทดสอบเชิงลึก (Mock Comprehensive Data)":
    df_cust, df_so, df_do, df_inv = generate_mock_datasets()
else:
    f_cust = st.sidebar.file_uploader("1. อัปโหลด Customer Master", type=["csv", "xlsx"])
    f_so = st.sidebar.file_uploader("2. อัปโหลด Sales Order (PO/SO)", type=["csv", "xlsx"])
    f_do = st.sidebar.file_uploader("3. อัปโหลด Delivery Order (DO)", type=["csv", "xlsx"])
    f_inv = st.sidebar.file_uploader("4. อัปโหลด Invoice / AR Ledger", type=["csv", "xlsx"])

    if f_cust and f_so and f_do and f_inv:
        df_cust = load_file(f_cust)
        df_so = load_file(f_so)
        df_do = load_file(f_do)
        df_inv = load_file(f_inv)
    else:
        st.info("กรุณาอัปโหลดไฟล์ให้ครบทั้ง 4 ตาราง เพื่อเริ่มการวิเคราะห์เชิงลึก หรือเลือก 'ใช้ข้อมูลทดสอบเชิงลึก'")
        st.stop()

# ==============================================================================
# 2. เครื่องมือประมวลผลและตรรกะการตรวจสอบ (Audit Engine)
# ==============================================================================
# รวมข้อมูล SO + DO + INV เข้ากับ Customer
m1 = pd.merge(df_so, df_cust, on="Customer_ID", how="left")
m2 = pd.merge(m1, df_do, on="SO_No", how="outer")
audit_master = pd.merge(m2, df_inv, on="SO_No", how="outer")

# คำนวณยอดเงิน มูลค่าสุทธิ และลูกหนี้คงเหลือ
audit_master["Gross_Amount"] = audit_master["Billed_Qty"] * audit_master["Billed_Price"]
audit_master["Discount_Amount"] = audit_master["Gross_Amount"] * (audit_master["Discount_Pct"].fillna(0) / 100.0)
audit_master["Net_Invoice_Amount"] = audit_master["Gross_Amount"] - audit_master["Discount_Amount"]
audit_master["Outstanding_AR"] = audit_master["Net_Invoice_Amount"] - audit_master["Paid_Amount"].fillna(0)
audit_master["Outstanding_AR"] = audit_master["Outstanding_AR"].apply(lambda x: max(x, 0.0) if pd.notna(x) else 0.0)

# วันที่ครบกำหนด (Due Date)
audit_master["INV_Date"] = pd.to_datetime(audit_master["INV_Date"])
audit_master["Effective_Term"] = audit_master["Approved_Term"].fillna(audit_master["Standard_Term"]).fillna(30)
audit_master["Due_Date"] = audit_master["INV_Date"] + pd.to_timedelta(audit_master["Effective_Term"], unit="D")

# กฎการประเมิน 3-Way Matching
def eval_3way(row):
    errs = []
    if pd.isna(row["DO_No"]) and pd.notna(row["INV_No"]):
        errs.append("ไม่มีใบส่งของ (DO ขาด) แต่เปิดบิลแล้ว เสี่ยงเป็นยอดขายทิพย์")
    if pd.notna(row["DO_No"]) and pd.isna(row["INV_No"]):
        errs.append("ส่งของแล้วแต่ยังไม่ได้ออกบิล (Unbilled Revenue)")
    if pd.notna(row["Delivered_Qty"]) and pd.notna(row["Billed_Qty"]):
        if row["Billed_Qty"] > row["Delivered_Qty"]:
            errs.append(f"เปิดบิลเกินยอดส่งจริง ({row['Billed_Qty']} > {row['Delivered_Qty']})")
    if pd.notna(row["Approved_Price"]) and pd.notna(row["Billed_Price"]):
        if row["Billed_Price"] > row["Approved_Price"]:
            errs.append(f"ราคาขายในบิลสูงกว่าที่อนุมัติใน SO ({row['Billed_Price']} > {row['Approved_Price']})")
    return "; ".join(errs) if errs else "ปกติ"

audit_master["3Way_Finding"] = audit_master.apply(eval_3way, axis=1)

# กฎการประเมิน Credit & Terms
def eval_credit(row):
    errs = []
    if pd.notna(row["Approved_Term"]) and pd.notna(row["Standard_Term"]):
        if row["Approved_Term"] > row["Standard_Term"]:
            errs.append(f"ให้เทอมเกินมาตรฐาน ({row['Approved_Term']} วัน > มาตรฐาน {row['Standard_Term']} วัน)")
    if pd.notna(row["Outstanding_AR"]) and pd.notna(row["Credit_Limit"]):
        if row["Outstanding_AR"] > row["Credit_Limit"]:
            errs.append(f"ยอดหนี้คงค้างเกินวงเงินสินเชื่อ ({row['Outstanding_AR']:,.0f} > วงเงิน {row['Credit_Limit']:,.0f})")
    return "; ".join(errs) if errs else "ปกติ"

audit_master["Credit_Finding"] = audit_master.apply(eval_credit, axis=1)

# กฎการตรวจจับ Revenue Leakage
def eval_leakage(row):
    errs = []
    # 1. ส่วนลดเกินอนุมัติ
    if pd.notna(row["Discount_Pct"]) and pd.notna(row["Max_Discount_Pct"]):
        if row["Discount_Pct"] > row["Max_Discount_Pct"]:
            errs.append(f"ให้ส่วนลดเกินเพดานอนุมัติ ({row['Discount_Pct']}% > อนุมัติ {row['Max_Discount_Pct']}%)")
    # 2. ยกเลิกบิลหลังปิดงวด
    if row.get("Status") == "Cancelled" and pd.notna(row.get("Cancelled_Date")):
        inv_month = row["INV_Date"].month if pd.notna(row["INV_Date"]) else None
        can_date = pd.to_datetime(row["Cancelled_Date"])
        if inv_month and can_date.month != inv_month:
            errs.append(f"ยกเลิกใบแจ้งหนี้ข้ามงวดบัญชี (ออกบิลเดือน {inv_month} ยกเลิกวันที่ {can_date.strftime('%Y-%m-%d')}) เสี่ยงตกแต่งรายได้")
    return "; ".join(errs) if errs else "ปกติ"

audit_master["Leakage_Finding"] = audit_master.apply(eval_leakage, axis=1)

# ==============================================================================
# 3. สรุปภาพรวมแดชบอร์ดผู้บริหาร (Executive KPIs)
# ==============================================================================
total_tx = len(audit_master)
c_3way = (audit_master["3Way_Finding"] != "ปกติ").sum()
c_credit = (audit_master["Credit_Finding"] != "ปกติ").sum()
c_leakage = (audit_master["Leakage_Finding"] != "ปกติ").sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("รายการตรวจทั้งหมด", f"{total_tx} รายการ")
kpi2.metric("3-Way Mismatch", f"{c_3way} ประเด็น", delta_color="inverse")
kpi3.metric("Credit & Term Breach", f"{c_credit} ประเด็น", delta_color="inverse")
kpi4.metric("Revenue Leakage", f"{c_leakage} ประเด็น", delta_color="inverse")

st.markdown("---")

# ==============================================================================
# 4. แท็บการตรวจสอบเฉพาะด้านทั้ง 6 มิติ
# ==============================================================================
t1, t2, t3, t4, t5, t6 = st.tabs([
    "1. 3-Way Matching",
    "2. Credit & Term Violations",
    "3. Ghost Customers & Circular",
    "4. Revenue Leakage",
    "5. Aging & ECL (TFRS 9)",
    "6. AI Audit Report (Gemini)"
])

# ----------------- TAB 1: 3-Way Matching -----------------
with t1:
    st.subheader("Automated 3-Way Matching (Sales Order - Delivery - Invoice)")
    ex_3way = audit_master[audit_master["3Way_Finding"] != "ปกติ"]
    cols_show = ["SO_No", "Customer_Name", "DO_No", "INV_No", "SO_Qty", "Delivered_Qty", "Billed_Qty", "Approved_Price", "Billed_Price", "3Way_Finding"]
    if not ex_3way.empty:
        st.dataframe(ex_3way[cols_show].fillna("-"), use_container_width=True)
    else:
        st.success("ไม่พบส่วนต่างในกระบวนการ 3-Way Matching")

# ----------------- TAB 2: Credit Violations -----------------
with t2:
    st.subheader("การอนุมัติวงเงินสินเชื่อและเงื่อนไขระยะเวลาชำระ (Credit Controls)")
    ex_credit = audit_master[audit_master["Credit_Finding"] != "ปกติ"]
    cols_credit = ["SO_No", "Customer_Name", "Credit_Limit", "Outstanding_AR", "Standard_Term", "Approved_Term", "Credit_Finding"]
    if not ex_credit.empty:
        st.dataframe(ex_credit[cols_credit].fillna("-"), use_container_width=True)
    else:
        st.success("การให้วงเงินและเครดิตเทอมเป็นไปตามระเบียบขององค์กร")

# ----------------- TAB 3: Ghost Customers & Circular -----------------
with t3:
    st.subheader("ตรวจสอบลูกหนี้ที่ไม่มีตัวตนจริงและนิติกรรมอำพราง (Ghost Customers)")
    st.caption("ตรวจสอบความเชื่อมโยงของเลขประจำตัวผู้เสียภาษี (Tax ID) และที่อยู่สถานประกอบการที่ซ้ำซ้อน")
    
    # เช็ก Tax ID ซ้ำ
    dup_tax = df_cust[df_cust.duplicated(subset=["Tax_ID"], keep=False)]
    # เช็ก Address ซ้ำ
    dup_addr = df_cust[df_cust.duplicated(subset=["Address"], keep=False)]
    
    c_tax, c_addr = st.columns(2)
    with c_tax:
        st.markdown("**1. ลูกค้าที่ใช้เลขประจำตัวผู้เสียภาษี (Tax ID) ซ้ำกัน:**")
        if not dup_tax.empty:
            st.dataframe(dup_tax[["Customer_ID", "Customer_Name", "Tax_ID"]], use_container_width=True)
        else:
            st.success("ไม่พบ Tax ID ซ้ำ")
            
    with c_addr:
        st.markdown("**2. ลูกค้าที่ใช้ที่อยู่จดทะเบียนเดียวกัน (Shared Address):**")
        if not dup_addr.empty:
            st.dataframe(dup_addr[["Customer_ID", "Customer_Name", "Address"]], use_container_width=True)
        else:
            st.success("ไม่พบที่อยู่ซ้ำซ้อน")

# ----------------- TAB 4: Revenue Leakage -----------------
with t4:
    st.subheader("การรั่วไหลของรายได้และการตกแต่งบัญชี (Revenue Leakage & Cut-off)")
    ex_leakage = audit_master[audit_master["Leakage_Finding"] != "ปกติ"]
    cols_leak = ["INV_No", "Customer_Name", "Gross_Amount", "Discount_Pct", "Max_Discount_Pct", "Status", "Cancelled_Date", "Leakage_Finding"]
    if not ex_leakage.empty:
        st.dataframe(ex_leakage[cols_leak].fillna("-"), use_container_width=True)
    else:
        st.success("ไม่พบข้อสังเกตเรื่องส่วนลดทับซ้อนหรือการยกเลิกบิลผิดปกติ")

# ----------------- TAB 5: Aging & TFRS 9 ECL -----------------
with t5:
    st.subheader("วิเคราะห์อายุหนี้และการประเมินผลขาดทุนด้านเครดิต (TFRS 9 ECL)")
    as_of = pd.to_datetime(date.today())
    audit_master["Overdue_Days"] = (as_of - audit_master["Due_Date"]).dt.days.fillna(0)
    
    def bucket_aging(d):
        if d <= 0: return "1. ยังไม่ถึงกำหนด (Current)"
        elif d <= 30: return "2. เกินกำหนด 1-30 วัน"
        elif d <= 60: return "3. เกินกำหนด 31-60 วัน"
        elif d <= 90: return "4. เกินกำหนด 61-90 วัน"
        else: return "5. เกินกำหนด > 90 วัน (NPL)"
        
    audit_master["Aging_Group"] = audit_master["Overdue_Days"].apply(bucket_aging)
    ar_open = audit_master[audit_master["Outstanding_AR"] > 0]
    
    aging_sum = ar_open.groupby("Aging_Group")["Outstanding_AR"].sum().reset_index()
    
    # อัตรา Loss Rate ตาม TFRS 9 Provision Matrix
    loss_rates = {
        "1. ยังไม่ถึงกำหนด (Current)": 0.005,
        "2. เกินกำหนด 1-30 วัน": 0.02,
        "3. เกินกำหนด 31-60 วัน": 0.05,
        "4. เกินกำหนด 61-90 วัน": 0.15,
        "5. เกินกำหนด > 90 วัน (NPL)": 0.50
    }
    
    aging_sum["Loss_Rate_%"] = aging_sum["Aging_Group"].map(lambda x: loss_rates.get(x, 0) * 100)
    aging_sum["ECL_Allowance"] = aging_sum["Aging_Group"].map(lambda x: loss_rates.get(x, 0)) * aging_sum["Outstanding_AR"]
    
    col_g, col_t = st.columns([1, 1])
    with col_g:
        fig_ar = px.bar(aging_sum, x="Aging_Group", y="Outstanding_AR", title="ยอดลูกหนี้คงค้างแยกตามชั้นอายุ", color="Aging_Group")
        st.plotly_chart(fig_ar, use_container_width=True)
    with col_t:
        st.markdown("**ตารางคำนวณ ECL Provision Matrix สิ้นงวด:**")
        st.dataframe(
            aging_sum.style.format({"Outstanding_AR": "{:,.2f}", "Loss_Rate_%": "{:.2f}%", "ECL_Allowance": "{:,.2f}"}),
            use_container_width=True
        )
        total_ecl = aging_sum["ECL_Allowance"].sum()
        st.metric("ประมาณการค่าเผื่อผลขาดทุนด้านเครดิต (ECL) รวม", f"{total_ecl:,.2f} บาท")

# ----------------- TAB 6: AI Audit Report -----------------
with t6:
    st.subheader("ร่างรายงานการตรวจสอบภายในอัตโนมัติด้วย Generative AI (Gemini)")
    
    # รวบรวมข้อผิดปกติทั้งหมด
    all_findings = []
    for idx, r in audit_master.iterrows():
        f = []
        if r["3Way_Finding"] != "ปกติ": f.append(f"[3-Way] {r['3Way_Finding']}")
        if r["Credit_Finding"] != "ปกติ": f.append(f"[Credit] {r['Credit_Finding']}")
        if r["Leakage_Finding"] != "ปกติ": f.append(f"[Leakage] {r['Leakage_Finding']}")
        if f:
            all_findings.append({
                "SO_No": r.get("SO_No", "-"),
                "Customer": r.get("Customer_Name", "-"),
                "Issues": " | ".join(f)
            })
    
    df_issues = pd.DataFrame(all_findings)
    st.write(f"พบรายการที่มีประเด็นความเสี่ยงรวมทั้งสิ้น: **{len(df_issues)} รายการ**")
    
    if st.button("กดให้ AI สรุปรายงานข้อตรวจพบและข้อเสนอแนะ", type="primary"):
        if not api_key:
            st.error("กรุณาระบุ Gemini API Key ที่แถบด้านซ้ายมือ")
        elif df_issues.empty:
            st.success("ไม่พบข้อผิดปกติในระบบ ข้อมูลมีความถูกต้องครบถ้วนสมบูรณ์")
        else:
            with st.spinner("Gemini กำลังสังเคราะห์ข้อมูลและร่างรายงานตรวจสอบตามมาตรฐานสากล..."):
                try:
                    client = genai.Client(api_key=api_key)
                    md_table = df_issues.to_markdown(index=False)
                    prompt = f"""
คุณเป็นหัวหน้าฝ่ายตรวจสอบภายใน (Chief Audit Executive) และผู้เชี่ยวชาญด้านบัญชี (TFRS 15, TFRS 9)
ได้รับข้อมูลข้อตรวจพบจากการทดสอบการควบคุมและการกระทบยอดข้อมูลในวงจรรายได้และลูกหนี้ ดังนี้:

{md_table}

กรุณาร่าง 'รายงานข้อตรวจพบและข้อเสนอแนะของการตรวจสอบภายใน' (Internal Audit Findings and Recommendations Report) โดยครอบคลุม:
1. **บทสรุปสำหรับผู้บริหาร (Executive Summary):** สรุปสภาพแวดล้อมการควบคุมภายใน และระดับความเสี่ยงของกระบวนการขาย
2. **การวิเคราะห์ข้อตรวจพบรายประเด็น (Detailed Findings):** แบ่งตามหมวดหมู่ (3-Way Matching, การให้สินเชื่อเกินอำนาจ, ความเสี่ยงลูกหนี้ทิพย์, และการรั่วไหลของรายได้)
   โดยในแต่ละหมวดให้วิเคราะห์ตามโครงสร้าง 5Cs:
   - สภาพการณ์ที่ตรวจพบ (Condition)
   - เกณฑ์มาตรฐาน/การควบคุมที่ควรมี (Criteria)
   - ผลกระทบและความเสี่ยงต่อองค์กร (Consequence/Risk)
   - สาเหตุที่เป็นไปได้ (Cause)
3. **ข้อเสนอแนะต่อฝ่ายบริหาร (Management Recommendations):** แนวทางการปรับปรุงการควบคุมภายในเชิงป้องกัน (Preventive Controls) ทั้งในระดับระบบ ERP และระดับนโยบาย
"""
                    res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.markdown("### ร่างรายงานการตรวจสอบภายในฉบับสมบูรณ์")
                    st.markdown(res.text)
                    
                    st.download_button(
                        label="ดาวน์โหลดรายงาน (.md)",
                        data=res.text,
                        file_name="Internal_Audit_Revenue_Report.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ API: {str(e)}")
