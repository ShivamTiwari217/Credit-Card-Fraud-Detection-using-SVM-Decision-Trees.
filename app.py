import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield | Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

#MainMenu, footer, header { visibility: hidden; }

.kpi-card {
    background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    transition: border-color 0.2s;
    margin-bottom: 8px;
}
.kpi-card:hover { border-color: #58a6ff; }
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    margin: 8px 0 4px;
}
.kpi-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.kpi-delta { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3;
    padding: 16px 0 8px;
    border-bottom: 1px solid #30363d;
    margin-bottom: 20px;
    letter-spacing: 0.01em;
}

.alert-fraud {
    background: linear-gradient(135deg, #3d1a1a, #2d1515);
    border: 1px solid #f85149;
    border-left: 4px solid #f85149;
    border-radius: 8px;
    padding: 16px 20px;
    color: #ffa198;
    font-weight: 500;
    margin-top: 12px;
}
.alert-legit {
    background: linear-gradient(135deg, #1a3d2a, #15302a);
    border: 1px solid #3fb950;
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 16px 20px;
    color: #7ee787;
    font-weight: 500;
    margin-top: 12px;
}

.brand {
    font-size: 1.3rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: -0.02em;
    padding: 8px 0 24px;
}
.brand span { color: #8b949e; font-weight: 400; }

.info-box {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.84rem;
    color: #8b949e;
    margin-bottom: 16px;
}

label { color: #8b949e !important; font-size: 0.82rem !important; }
button[data-baseweb="tab"] { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; color: #8b949e !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    data = joblib.load("fraud_model.pkl")
    return data["model"], data["scaler"], data["threshold"]

try:
    model, scaler, saved_threshold = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model: {e}")
    saved_threshold = 0.5

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">🛡️ FraudShield <span>v1.0</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**⚙️ Detection Settings**")

    threshold = st.slider(
        "Risk Threshold", 0.0, 1.0,
        value=float(saved_threshold), step=0.01,
        help="Lower = catch more fraud (higher false positives). Higher = fewer alerts."
    )

    if threshold < 0.3:
        st.caption("🔴 Very Aggressive — high false positive rate")
    elif threshold < 0.5:
        st.caption("🟠 Aggressive — favours recall")
    elif threshold < 0.7:
        st.caption("🟡 Balanced")
    else:
        st.caption("🟢 Conservative — favours precision")

    if model_loaded:
        st.caption(f"Model-tuned threshold: `{saved_threshold:.4f}`")

    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.caption(
        "FraudShield uses an XGBoost model trained on credit card transaction data "
        "with SMOTE oversampling and an optimised classification threshold."
    )
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y')}")

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 28px'>
    <h1 style='font-size:1.9rem; font-weight:700; color:#e6edf3; margin:0; letter-spacing:-0.02em'>
        Credit Card Fraud Detection
    </h1>
    <p style='color:#8b949e; font-size:0.93rem; margin:6px 0 0'>
        Real-time risk scoring · Batch analysis · Model diagnostics · Threshold optimisation
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Single Transaction",
    "📂  Batch Analysis",
    "📊  Model Performance",
    "📈  Risk Intelligence"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Transaction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Transaction Risk Scorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Enter the 30 normalised transaction features (V1–V28 + Amount + Time) to get an instant fraud risk score.</div>', unsafe_allow_html=True)

    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        features = []

        with col_a:
            st.markdown("**Features 1–15**")
            for name in feature_names[:15]:
                features.append(st.number_input(name, value=0.0, format="%.6f", key=f"f_{name}"))

        with col_b:
            st.markdown("**Features 16–30**")
            for name in feature_names[15:]:
                features.append(st.number_input(name, value=0.0, format="%.6f", key=f"f_{name}"))

        submitted = st.form_submit_button("🔍  Analyse Transaction", use_container_width=True)

    if submitted and model_loaded:
        arr      = np.array(features).reshape(1, -1)
        scaled   = scaler.transform(arr)
        prob     = model.predict_proba(scaled)[0][1]
        pred     = 1 if prob >= threshold else 0

        color   = "#f85149" if pred == 1 else "#3fb950"
        verdict = "FRAUDULENT" if pred == 1 else "LEGITIMATE"
        risk    = "HIGH" if prob > 0.7 else "MEDIUM" if prob > threshold else "LOW"
        r_color = "#f85149" if risk == "HIGH" else "#d29922" if risk == "MEDIUM" else "#3fb950"

        r1, r2, r3 = st.columns(3)
        for col, val, label, clr in [
            (r1, verdict,      "Verdict",          color),
            (r2, f"{prob:.2%}","Fraud Probability", color),
            (r3, risk,         "Risk Level",        r_color),
        ]:
            with col:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{clr}; font-size:1.5rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 30, "color": "#e6edf3", "family": "DM Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
                "bar": {"color": color},
                "bgcolor": "#1c2128",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0,  30], "color": "#1a3d2a"},
                    {"range": [30, 60], "color": "#2d2a1a"},
                    {"range": [60,100], "color": "#3d1a1a"},
                ],
                "threshold": {"line": {"color": "#58a6ff", "width": 3},
                              "thickness": 0.85, "value": threshold * 100}
            },
            title={"text": "Fraud Risk Score", "font": {"color": "#8b949e", "size": 13}}
        ))
        fig_g.update_layout(paper_bgcolor="#0d1117", font_color="#e6edf3",
                            height=270, margin=dict(t=40, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        if pred == 1:
            st.markdown(f'<div class="alert-fraud">🚨 <strong>FRAUD ALERT</strong> — This transaction scores <strong>{prob:.2%}</strong> fraud probability, exceeding the <strong>{threshold:.2f}</strong> threshold. Recommend immediate review and potential card block.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-legit">✅ <strong>TRANSACTION CLEARED</strong> — Risk score of <strong>{prob:.2%}</strong> is below the threshold of <strong>{threshold:.2f}</strong>. Transaction appears legitimate.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Batch Transaction Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Upload a CSV of transactions (same feature structure as training data). The model will score every row and provide a downloadable risk report.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None and model_loaded:
        data = pd.read_csv(uploaded_file)

        with st.spinner("Scoring transactions..."):
            scaled_data   = scaler.transform(data)
            probabilities = model.predict_proba(scaled_data)[:, 1]
            predictions   = (probabilities >= threshold).astype(int)

        data["Fraud_Probability"] = probabilities.round(4)
        data["Prediction"]        = predictions
        data["Risk_Level"]        = pd.cut(
            probabilities,
            bins=[0, 0.3, threshold, 0.7, 1.0],
            labels=["Low", "Medium", "Elevated", "High"],
            include_lowest=True
        )

        total      = len(data)
        fraud_cnt  = predictions.sum()
        avg_prob   = probabilities.mean() * 100
        high_risk  = (probabilities > 0.7).sum()

        k1, k2, k3, k4 = st.columns(4)
        for col, val, label, clr, delta in [
            (k1, str(total),           "Total Transactions",  "#58a6ff", ""),
            (k2, str(fraud_cnt),       "Flagged as Fraud",    "#f85149", f"{fraud_cnt/total*100:.1f}% of total"),
            (k3, f"{avg_prob:.1f}%",   "Avg Risk Score",      "#d29922", ""),
            (k4, str(high_risk),       "High Risk (>70%)",    "#f85149", "Needs immediate review"),
        ]:
            with col:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{clr}">{val}</div>
                    <div class="kpi-delta">{delta}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(x=probabilities[predictions==0], name="Legitimate",
                                         marker_color="#3fb950", opacity=0.7, nbinsx=40))
            fig_h.add_trace(go.Histogram(x=probabilities[predictions==1], name="Flagged",
                                         marker_color="#f85149", opacity=0.8, nbinsx=40))
            fig_h.add_vline(x=threshold, line_dash="dash", line_color="#58a6ff",
                            annotation_text=f"Threshold {threshold:.2f}", annotation_font_color="#58a6ff")
            fig_h.update_layout(
                title="Probability Distribution", barmode="overlay",
                paper_bgcolor="#0d1117", plot_bgcolor="#1c2128", font_color="#e6edf3",
                height=320, legend=dict(bgcolor="#0d1117"),
                xaxis=dict(title="Fraud Probability", gridcolor="#30363d"),
                yaxis=dict(title="Count", gridcolor="#30363d"),
                margin=dict(t=45, b=40, l=40, r=20)
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with c2:
            rc = data["Risk_Level"].value_counts()
            fig_d = go.Figure(go.Pie(
                labels=rc.index, values=rc.values, hole=0.6,
                marker_colors=["#3fb950","#d29922","#e3b341","#f85149"],
                textfont=dict(color="#e6edf3")
            ))
            fig_d.update_layout(
                title="Risk Level Breakdown",
                paper_bgcolor="#0d1117", font_color="#e6edf3",
                height=320, legend=dict(bgcolor="#0d1117"),
                margin=dict(t=45, b=20)
            )
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown('<div class="section-header">Transaction Results</div>', unsafe_allow_html=True)
        filt = st.radio("Show:", ["All", "Fraud Only", "Legitimate Only"], horizontal=True, label_visibility="collapsed")
        df_show = data if filt == "All" else data[data["Prediction"] == (1 if filt == "Fraud Only" else 0)]
        st.dataframe(df_show[["Fraud_Probability","Prediction","Risk_Level"]].head(300), use_container_width=True, height=300)

        st.download_button(
            "⬇️  Download Full Risk Report (CSV)",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=f"fraud_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Upload <code>X_test.csv</code> and <code>y_test.csv</code> to evaluate performance at your chosen threshold.</div>', unsafe_allow_html=True)

    cx, cy = st.columns(2)
    with cx: xf = st.file_uploader("X_test.csv", type=["csv"], key="xtest")
    with cy: yf = st.file_uploader("y_test.csv", type=["csv"], key="ytest")

    if xf and yf and model_loaded:
        X_t  = pd.read_csv(xf)
        y_t  = pd.read_csv(yf).values.ravel()
        Xs   = scaler.transform(X_t)
        yp   = model.predict_proba(Xs)[:, 1]
        yprd = (yp >= threshold).astype(int)
        cm   = confusion_matrix(y_t, yprd)
        tn, fp, fn, tp_ = cm.ravel()

        prec = tp_ / (tp_ + fp) if (tp_ + fp) > 0 else 0
        rec  = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
        fpr_a, tpr_a, _ = roc_curve(y_t, yp)
        ra   = auc(fpr_a, tpr_a)
        pa, re_a, _ = precision_recall_curve(y_t, yp)
        pra  = auc(re_a, pa)

        m1,m2,m3,m4,m5 = st.columns(5)
        for col, val, label, clr in [
            (m1, f"{prec:.2%}",  "Precision", "#58a6ff"),
            (m2, f"{rec:.2%}",   "Recall",    "#3fb950"),
            (m3, f"{f1:.2%}",    "F1 Score",  "#d29922"),
            (m4, f"{ra:.4f}",    "ROC-AUC",   "#a371f7"),
            (m5, f"{pra:.4f}",   "PR-AUC",    "#f78166"),
        ]:
            with col:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{clr}; font-size:1.4rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)

        with p1:
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp],[fn, tp_]],
                x=["Predicted: Legit","Predicted: Fraud"],
                y=["Actual: Legit","Actual: Fraud"],
                colorscale=[[0,"#1c2128"],[1,"#58a6ff"]],
                text=[[f"TN: {tn:,}",f"FP: {fp:,}"],[f"FN: {fn:,}",f"TP: {tp_:,}"]],
                texttemplate="%{text}", textfont={"size":16,"color":"white"}, showscale=False
            ))
            fig_cm.update_layout(title="Confusion Matrix", paper_bgcolor="#0d1117",
                                 plot_bgcolor="#1c2128", font_color="#e6edf3",
                                 height=340, margin=dict(t=50,b=40,l=20,r=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with p2:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_a, y=tpr_a, mode="lines",
                                         name=f"AUC={ra:.3f}", line=dict(color="#58a6ff", width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                         line=dict(color="#8b949e", dash="dash", width=1), showlegend=False))
            fig_roc.update_layout(
                title="ROC Curve",
                xaxis=dict(title="False Positive Rate", gridcolor="#30363d"),
                yaxis=dict(title="True Positive Rate",  gridcolor="#30363d"),
                paper_bgcolor="#0d1117", plot_bgcolor="#1c2128", font_color="#e6edf3",
                height=340, legend=dict(bgcolor="#0d1117"), margin=dict(t=50,b=40,l=40,r=20)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        p3, p4 = st.columns(2)
        with p3:
            fig_pr = go.Figure(go.Scatter(x=re_a, y=pa, mode="lines",
                                          name=f"PR-AUC={pra:.3f}", line=dict(color="#3fb950", width=2.5)))
            fig_pr.update_layout(
                title="Precision-Recall Curve",
                xaxis=dict(title="Recall",    gridcolor="#30363d"),
                yaxis=dict(title="Precision", gridcolor="#30363d"),
                paper_bgcolor="#0d1117", plot_bgcolor="#1c2128", font_color="#e6edf3",
                height=320, legend=dict(bgcolor="#0d1117"), margin=dict(t=50,b=40,l=40,r=20)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        with p4:
            fig_sd = go.Figure()
            fig_sd.add_trace(go.Histogram(x=yp[y_t==0], name="Legitimate",
                                          marker_color="#3fb950", opacity=0.7, nbinsx=50))
            fig_sd.add_trace(go.Histogram(x=yp[y_t==1], name="Fraud",
                                          marker_color="#f85149", opacity=0.8, nbinsx=50))
            fig_sd.add_vline(x=threshold, line_dash="dash", line_color="#58a6ff",
                             annotation_text="Threshold", annotation_font_color="#58a6ff")
            fig_sd.update_layout(
                title="Score Distribution by Class", barmode="overlay",
                xaxis=dict(title="Fraud Probability", gridcolor="#30363d"),
                yaxis=dict(title="Count (log)", gridcolor="#30363d", type="log"),
                paper_bgcolor="#0d1117", plot_bgcolor="#1c2128", font_color="#e6edf3",
                height=320, legend=dict(bgcolor="#0d1117"), margin=dict(t=50,b=40,l=40,r=20)
            )
            st.plotly_chart(fig_sd, use_container_width=True)
    else:
        st.markdown('<div class="info-box" style="text-align:center;padding:48px">📊 Upload X_test.csv and y_test.csv above to view charts</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Risk Intelligence (NEW)
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Threshold Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Explore how changing the threshold affects precision, recall, F1, and your bottom line. Upload test data to begin.</div>', unsafe_allow_html=True)

    ri_x = st.file_uploader("X_test.csv", type=["csv"], key="ri_x")
    ri_y = st.file_uploader("y_test.csv", type=["csv"], key="ri_y")

    if ri_x and ri_y and model_loaded:
        Xr   = scaler.transform(pd.read_csv(ri_x))
        yr   = pd.read_csv(ri_y).values.ravel()
        pr_  = model.predict_proba(Xr)[:, 1]

        ts   = np.arange(0.01, 0.99, 0.01)
        precs, recs, f1s, fprs = [], [], [], []
        for t in ts:
            pd_ = (pr_ >= t).astype(int)
            tp_ = ((pd_==1)&(yr==1)).sum(); fp_ = ((pd_==1)&(yr==0)).sum()
            fn_ = ((pd_==0)&(yr==1)).sum(); tn_ = ((pd_==0)&(yr==0)).sum()
            p   = tp_/(tp_+fp_) if (tp_+fp_)>0 else 0
            r   = tp_/(tp_+fn_) if (tp_+fn_)>0 else 0
            precs.append(p); recs.append(r)
            f1s.append(2*p*r/(p+r) if (p+r)>0 else 0)
            fprs.append(fp_/(fp_+tn_) if (fp_+tn_)>0 else 0)

        fig_s = go.Figure()
        for vals, name, clr, dash in [
            (precs, "Precision", "#58a6ff", "solid"),
            (recs,  "Recall",    "#3fb950", "solid"),
            (f1s,   "F1 Score",  "#d29922", "solid"),
            (fprs,  "False Positive Rate", "#f85149", "dot"),
        ]:
            fig_s.add_trace(go.Scatter(x=ts, y=vals, name=name, mode="lines",
                                       line=dict(color=clr, width=2, dash=dash)))
        fig_s.add_vline(x=threshold, line_dash="dash", line_color="#a371f7", line_width=2,
                        annotation_text=f"Current: {threshold:.2f}", annotation_font_color="#a371f7")
        fig_s.update_layout(
            title="Precision / Recall / F1 / FPR across Thresholds",
            xaxis=dict(title="Classification Threshold", gridcolor="#30363d"),
            yaxis=dict(title="Score", gridcolor="#30363d", range=[0,1]),
            paper_bgcolor="#0d1117", plot_bgcolor="#1c2128", font_color="#e6edf3",
            height=420, legend=dict(bgcolor="#0d1117"), margin=dict(t=50,b=40,l=40,r=20)
        )
        st.plotly_chart(fig_s, use_container_width=True)

        # ── Business Cost Simulator ───────────────────────────────────────
        st.markdown('<div class="section-header">💼 Business Cost Simulator</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Model the financial impact of missed fraud vs. unnecessary investigations at your current threshold.</div>', unsafe_allow_html=True)

        bc1, bc2 = st.columns(2)
        with bc1: cost_fn = st.number_input("Cost per missed fraud (₹)", value=10000, step=500)
        with bc2: cost_fp = st.number_input("Cost per false investigation (₹)", value=500, step=100)

        curr = (pr_ >= threshold).astype(int)
        tp_c = ((curr==1)&(yr==1)).sum()
        fp_c = ((curr==1)&(yr==0)).sum()
        fn_c = ((curr==0)&(yr==1)).sum()

        bc3, bc4, bc5 = st.columns(3)
        for col, val, label, clr, sub in [
            (bc3, f"₹{fn_c*cost_fn:,.0f}", "Missed Fraud Cost",       "#f85149", f"{fn_c} cases × ₹{cost_fn:,}"),
            (bc4, f"₹{fp_c*cost_fp:,.0f}", "False Investigation Cost", "#d29922", f"{fp_c} cases × ₹{cost_fp:,}"),
            (bc5, f"₹{fn_c*cost_fn+fp_c*cost_fp:,.0f}", "Total Estimated Cost", "#e6edf3", f"At threshold {threshold:.2f}"),
        ]:
            with col:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{clr}; font-size:1.4rem">{val}</div>
                    <div class="kpi-delta">{sub}</div>
                </div>""", unsafe_allow_html=True)

        # Cost curve across thresholds
        costs = [(((pr_>=t).astype(int)==0)&(yr==1)).sum()*cost_fn +
                 ((((pr_>=t).astype(int)==1)&(yr==0)).sum()*cost_fp) for t in ts]
        fig_cost = go.Figure(go.Scatter(x=ts, y=costs, mode="lines",
                                        line=dict(color="#f78166", width=2.5)))
        fig_cost.add_vline(x=threshold, line_dash="dash", line_color="#a371f7", line_width=2,
                           annotation_text=f"Current: {threshold:.2f}", annotation_font_color="#a371f7")
        fig_cost.update_layout(
            title="Total Business Cost vs. Threshold",
            xaxis=dict(title="Classification Threshold", gridcolor="#30363d"),
            yaxis=dict(title="Estimated Cost (₹)", gridcolor="#30363d"),
            paper_bgcolor="#0d1117", plot_bgcolor="#1c2128", font_color="#e6edf3",
            height=340, margin=dict(t=50, b=40, l=60, r=20)
        )
        st.plotly_chart(fig_cost, use_container_width=True)
        st.caption("💡 Drag the threshold slider in the sidebar to find the cost-minimising operating point.")

    else:
        st.markdown('<div class="info-box" style="text-align:center;padding:48px">📈 Upload X_test.csv and y_test.csv above to run sensitivity analysis</div>', unsafe_allow_html=True)