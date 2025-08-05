"""
Test User-Friendly Interface - Minimal Version
"""
import streamlit as st

st.set_page_config(
    page_title="ECG Heart Attack Detection - Test",
    page_icon="❤️",
    layout="wide"
)

def main():
    st.markdown("# ❤️ ECG Heart Attack Detection - Test")
    st.markdown("### This is a test of the user-friendly interface")
    
    st.success("✅ Interface loading successfully!")
    
    st.markdown("""
    ### 🧪 Test Components:
    
    - Streamlit configuration: ✅ Working
    - Page layout: ✅ Working  
    - Basic markdown: ✅ Working
    - Imports: ✅ Working
    """)
    
    if st.button("Test Button"):
        st.balloons()
        st.success("Button clicked successfully!")

if __name__ == "__main__":
    main()