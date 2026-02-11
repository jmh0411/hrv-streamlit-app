import numpy as np
import scipy
import streamlit as st

st.write("numpy version:", np.__version__)
st.write("scipy version:", scipy.__version__)

x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 4, 9])
area = np.trapz(y, x)
st.write("trapz 계산 결과:", area)
