import streamlit as st
from database import create_user, verify_user, log_activity
import re

# ---------------- Validators ---------------- #
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    return True, "Valid"

# ---------------- Auth UI ---------------- #
def show_login_page():
    st.markdown("""
    <style>
        .auth-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 40px;
        }
        .auth-title {
            text-align: center;
            color: #0e76a8;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<p class="auth-title">🏦 Loan Predictor</p>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

        # ---------------- Login Tab ---------------- #
        with tab1:
            st.markdown("### Welcome Back!")

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_btn = st.form_submit_button("🚀 Login")

            if login_btn:
                if not username.strip() or not password.strip():
                    st.error("Please fill in all fields")
                else:
                    user = verify_user(username.strip(), password.strip())
                    if user:
                        st.session_state['authenticated'] = True
                        st.session_state['user'] = user
                        log_activity(user['user_id'], 'LOGIN', "User logged in")
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

            if st.button("Forgot?"):
                st.info("Contact admin for password reset")

        # ---------------- Signup Tab ---------------- #
        with tab2:
            st.markdown("### Create New Account")

            with st.form("signup_form"):
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="Enter your email")
                username_signup = st.text_input("Username", placeholder="Choose a username")
                password_signup = st.text_input("Password", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                signup_btn = st.form_submit_button("📝 Create Account")

            if signup_btn:
                # Strip whitespace
                full_name = full_name.strip()
                email = email.strip()
                username_signup = username_signup.strip()
                password_signup = password_signup.strip()
                confirm_password = confirm_password.strip()

                if not full_name or not email or not username_signup or not password_signup or not confirm_password:
                    st.error("Please fill in all fields")
                elif not validate_email(email):
                    st.error("Invalid email format")
                elif password_signup != confirm_password:
                    st.error("Passwords do not match")
                else:
                    valid, msg = validate_password(password_signup)
                    if not valid:
                        st.error(msg)
                    else:
                        success, message = create_user(username_signup, email, password_signup, full_name)
                        if success:
                            st.success(message)
                            st.info("Please login with your credentials")
                        else:
                            st.error(message)


# ---------------- Logout ---------------- #
def logout():
    if 'user' in st.session_state:
        log_activity(st.session_state['user']['user_id'], 'LOGOUT', "User logged out")
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
    st.rerun()
