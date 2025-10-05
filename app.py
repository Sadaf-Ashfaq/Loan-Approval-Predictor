import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from auth import show_login_page, logout
from database import save_loan_application, get_user_applications, get_user_stats, log_activity

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None

@st.cache_resource
def load_models():
    model = joblib.load('loan_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, label_encoders, target_encoder, feature_names

@st.cache_data
def load_data():
    df = pd.read_csv('loan_approval.csv')
    df.columns = df.columns.str.strip().str.lower()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df

if not st.session_state['authenticated']:
    show_login_page()
else:
    model, label_encoders, target_encoder, feature_names = load_models()
    df = load_data()
    user = st.session_state['user']
    
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e1e1e 0%, #0e76a8 100%);
            padding: 15px 40px;
            margin: -80px -80px 30px -80px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo-text {
            font-size: 24px;
            font-weight: bold;
        }
        .user-info {
            font-size: 14px;
            color: #e0e0e0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #1e1e1e;
            padding: 0;
            border-radius: 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            padding: 0px 40px;
            background-color: #1e1e1e;
            border-radius: 0;
            color: #ffffff;
            font-size: 16px;
            font-weight: 500;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0e76a8;
            color: white;
            border-bottom: 3px solid #00d4ff;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #2a2a2a;
        }
        div[data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col_header1, col_header2, col_header3 = st.columns([2, 4, 2])
    
    with col_header1:
        st.markdown('<div style="font-size: 28px; font-weight: bold; color: #0e76a8;">🏦 Loan Predictor</div>', unsafe_allow_html=True)
    
    with col_header2:
        st.markdown("")
    
    with col_header3:
        col_user, col_logout = st.columns([3, 1])
        with col_user:
            st.markdown(f'<div style="text-align: right; padding-top: 10px;">👤 <b>{user["full_name"]}</b></div>', unsafe_allow_html=True)
        with col_logout:
            if st.button("Logout", type="secondary"):
                logout()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 New Application", "📊 Model Performance", "📈 Data Analytics", "📋 My Applications"])
    
    with tab1:
        st.title("💰 Loan Approval Prediction")
        st.markdown("### Enter Applicant Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 Personal Information")
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
            education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
            self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
        
        with col2:
            st.subheader("💵 Financial Information")
            income_annum = st.number_input("Annual Income ($)", min_value=0)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0)
            loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
        
        with col3:
            st.subheader("🏠 Assets Information")
            residential_assets_value = st.number_input("Residential Assets ($)", min_value=0)
            commercial_assets_value = st.number_input("Commercial Assets ($)", min_value=0)
            luxury_assets_value = st.number_input("Luxury Assets ($)", min_value=0)
            bank_asset_value = st.number_input("Bank Assets ($)", min_value=0)
        
        if st.button("🔮 Check Your Eligibility", type="primary"):
            input_data = {
                'no_of_dependents': no_of_dependents,
                'education': education,
                'self_employed': self_employed,
                'income_annum': income_annum,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'cibil_score': cibil_score,
                'residential_assets_value': residential_assets_value,
                'commercial_assets_value': commercial_assets_value,
                'luxury_assets_value': luxury_assets_value,
                'bank_asset_value': bank_asset_value
            }
            
            input_df = pd.DataFrame([input_data])
            for col in input_df.select_dtypes(include=['object']).columns:
                input_df[col] = input_df[col].str.strip()
            
            for col in label_encoders:
                if col in input_df.columns:
                    input_df[col] = label_encoders[col].transform(input_df[col])
            
            input_df = input_df[feature_names]
            
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)
            result = target_encoder.inverse_transform(prediction)[0]
            
            classes = list(target_encoder.classes_)
            proba_values = proba[0]
            
            if classes[0] == 'Approved':
                approval_prob = proba_values[0] * 100
                rejection_prob = proba_values[1] * 100
            else:
                approval_prob = proba_values[1] * 100
                rejection_prob = proba_values[0] * 100
            
            save_loan_application(user['user_id'], input_data, result, approval_prob, rejection_prob)
            log_activity(user['user_id'], 'PREDICTION', f"Loan prediction: {result}")
            
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if result == 'Approved':
                    st.success("### ✅ LOAN APPROVED")
                    st.balloons()
                else:
                    st.error("### ❌ LOAN REJECTED")
            
            with col_b:
                st.metric("Approval Probability", f"{approval_prob:.2f}%", 
                         delta=f"{approval_prob - 50:.1f}%" if approval_prob > 50 else None)
            
            with col_c:
                st.metric("Rejection Probability", f"{rejection_prob:.2f}%",
                         delta=f"{rejection_prob - 50:.1f}%" if rejection_prob > 50 else None,
                         delta_color="inverse")
            
            col_gauge1, col_gauge2 = st.columns(2)
            
            with col_gauge1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=approval_prob,
                    title={'text': "Approval Confidence"},
                    number={'suffix': "%"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [
                               {'range': [0, 30], 'color': "#ffcccc"},
                               {'range': [30, 70], 'color': "#fff4cc"},
                               {'range': [70, 100], 'color': "#ccffcc"}],
                           'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 70}}))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_gauge2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rejection_prob,
                    title={'text': "Rejection Risk"},
                    number={'suffix': "%"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkred"},
                           'steps': [
                               {'range': [0, 30], 'color': "#ccffcc"},
                               {'range': [30, 70], 'color': "#fff4cc"},
                               {'range': [70, 100], 'color': "#ffcccc"}],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}}))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🎯 Key Factors Influencing Decision")
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                importances = np.abs(model.coef_[0])
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(5)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title="Top 5 Influential Features",
                         color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            insights = []
            rejection_reasons = []
            
            if cibil_score >= 750:
                insights.append("✅ Excellent CIBIL score increases approval chances")
            elif cibil_score >= 650:
                insights.append("⚠️ Good CIBIL score, but could be better")
            else:
                insights.append("❌ Low CIBIL score significantly reduces approval chances")
                rejection_reasons.append(f"CIBIL Score ({cibil_score}) is below acceptable threshold (650+)")
            
            if income_annum >= 8000000:
                insights.append("✅ High annual income is favorable")
            elif income_annum >= 5000000:
                insights.append("⚠️ Moderate income level")
            else:
                insights.append("❌ Income may be insufficient for loan amount")
                rejection_reasons.append(f"Annual Income (${income_annum/1000000:.1f}M) is relatively low")
            
            total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
            if total_assets >= 20000000:
                insights.append("✅ Strong asset base supports application")
            elif total_assets >= 10000000:
                insights.append("⚠️ Moderate asset portfolio")
            else:
                insights.append("❌ Limited assets to secure the loan")
                rejection_reasons.append(f"Total Assets (${total_assets/1000000:.1f}M) are insufficient")
            
            loan_to_income_ratio = loan_amount / income_annum
            if loan_to_income_ratio > 5:
                insights.append("❌ Loan amount is very high relative to income")
                rejection_reasons.append(f"Loan-to-Income ratio ({loan_to_income_ratio:.1f}x) exceeds safe limit (5x)")
            elif loan_to_income_ratio > 3:
                insights.append("⚠️ Loan amount is moderately high compared to income")
            else:
                insights.append("✅ Loan amount is reasonable for your income")
            
            if education == 'Not Graduate':
                rejection_reasons.append("Educational qualification is below preferred level")
            
            if self_employed == 'Yes':
                rejection_reasons.append("Self-employment adds additional risk assessment requirements")
            
            if loan_term > 240:
                rejection_reasons.append(f"Loan term ({loan_term} months) is unusually long")
            
            if no_of_dependents >= 4:
                rejection_reasons.append(f"High number of dependents ({no_of_dependents}) increases financial burden")
            
            st.markdown("### 💡 Application Insights")
            for insight in insights:
                st.markdown(f"- {insight}")
            
            if result == 'Rejected' and rejection_reasons:
                st.markdown("### ⚠️ Primary Reasons for Rejection")
                st.error("Your application was rejected due to the following factors:")
                for i, reason in enumerate(rejection_reasons, 1):
                    st.markdown(f"{i}. **{reason}**")
                
                st.markdown("### 📋 Recommendations to Improve")
                recommendations = []
                if cibil_score < 650:
                    recommendations.append("🎯 Improve your CIBIL score by paying bills on time and reducing existing debt")
                if income_annum < 5000000:
                    recommendations.append("🎯 Consider co-applicant with additional income or wait for salary increment")
                if total_assets < 10000000:
                    recommendations.append("🎯 Build your asset base before reapplying")
                if loan_to_income_ratio > 3:
                    recommendations.append("🎯 Apply for a lower loan amount or increase down payment")
                
                for rec in recommendations:
                    st.info(rec)
            
            elif result == 'Approved':
                st.success("### 🎉 Congratulations! Your loan application meets all criteria")
                st.markdown("**Next Steps:**")
                st.markdown("- 📄 Submit required documentation")
                st.markdown("- 🔍 Complete verification process")
                st.markdown("- ✍️ Sign loan agreement")
                st.markdown("- 💰 Loan disbursement within 7-10 business days")
    
    with tab2:
        st.title("📊 Model Performance Metrics")
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        data = df.copy()
        le_target = LabelEncoder()
        
        categorical_features = ['education', 'self_employed']
        for col in categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        
        data['loan_status'] = le_target.fit_transform(data['loan_status'])
        
        if 'loan_id' in data.columns:
            data = data.drop('loan_id', axis=1)
        
        X = data.drop('loan_status', axis=1)
        y = data['loan_status']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        with col2:
            st.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
        with col3:
            st.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
        with col4:
            st.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=['Rejected', 'Approved'], y=['Rejected', 'Approved'],
                            color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='ROC Curve')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    
    with tab3:
        st.title("📈 Data Insights & Analysis")
        
        data = df.copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Applications", len(data))
        with col2:
            approval_rate = (data['loan_status'] == 'Approved').sum() / len(data) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        with col3:
            avg_loan = data['loan_amount'].mean()
            st.metric("Avg Loan Amount", f"${avg_loan/1000000:.1f}M")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Approval Rate by Education")
            edu_approval = data.groupby('education')['loan_status'].apply(lambda x: (x == 'Approved').sum() / len(x) * 100).reset_index()
            edu_approval.columns = ['Education', 'Approval Rate (%)']
            fig = px.bar(edu_approval, x='Education', y='Approval Rate (%)', color='Approval Rate (%)',
                         color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("Approval Rate by Employment")
            emp_approval = data.groupby('self_employed')['loan_status'].apply(lambda x: (x == 'Approved').sum() / len(x) * 100).reset_index()
            emp_approval.columns = ['Self Employed', 'Approval Rate (%)']
            fig = px.bar(emp_approval, x='Self Employed', y='Approval Rate (%)', color='Approval Rate (%)',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            st.subheader("Income Distribution by Status")
            fig = px.box(data, x='loan_status', y='income_annum', color='loan_status',
                         labels={'income_annum': 'Annual Income ($)', 'loan_status': 'Loan Status'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col_d:
            st.subheader("CIBIL Score Distribution")
            fig = px.histogram(data, x='cibil_score', color='loan_status', nbins=30,
                              labels={'cibil_score': 'CIBIL Score'}, barmode='overlay', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Loan Amount vs Income by Status")
        fig = px.scatter(data, x='income_annum', y='loan_amount', color='loan_status',
                         size='cibil_score', hover_data=['education', 'self_employed'],
                         labels={'income_annum': 'Annual Income ($)', 'loan_amount': 'Loan Amount ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.title("📋 My Loan Applications")
        
        user_apps = get_user_applications(user['user_id'])
        user_stats = get_user_stats(user['user_id'])
        
        if user_stats and user_stats['total_applications'] > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Applications", user_stats['total_applications'])
            with col2:
                st.metric("Approved", user_stats['approved'], delta=f"{user_stats['approved']/user_stats['total_applications']*100:.1f}%")
            with col3:
                st.metric("Rejected", user_stats['rejected'], delta=f"{user_stats['rejected']/user_stats['total_applications']*100:.1f}%", delta_color="inverse")
            with col4:
                st.metric("Avg Approval Rate", f"{user_stats['avg_approval_rate']:.1f}%")
            
            st.markdown("---")
            st.subheader("📊 Application History")
            
            if len(user_apps) > 0:
                display_df = user_apps[['application_id', 'income_annum', 'loan_amount', 'cibil_score', 
                                       'prediction', 'approval_probability', 'created_at']].copy()
                display_df.columns = ['ID', 'Income', 'Loan Amount', 'CIBIL', 'Status', 'Approval %', 'Date']
                display_df['Income'] = display_df['Income'].apply(lambda x: f"${x/1000000:.1f}M")
                display_df['Loan Amount'] = display_df['Loan Amount'].apply(lambda x: f"${x/1000000:.1f}M")
                display_df['Approval %'] = display_df['Approval %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("Status Distribution")
                    status_counts = user_apps['prediction'].value_counts()
                    fig = px.pie(values=status_counts.values, names=status_counts.index,
                                color=status_counts.index,
                                color_discrete_map={'Approved': '#28a745', 'Rejected': '#dc3545'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_chart2:
                    st.subheader("Approval Probability Trend")
                    fig = px.line(user_apps, x='created_at', y='approval_probability',
                                 markers=True, labels={'created_at': 'Date', 'approval_probability': 'Approval %'})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No applications found. Submit your first application in the 'New Application' tab!")
        else:
            st.info("🎯 You haven't submitted any loan applications yet. Go to 'New Application' tab to get started!")