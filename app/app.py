"""practice creating a streamlit app
must have streamlit pip-installed and active in the terminal
then run the following (assuming the cwd is the project root):
    streamlit run app/app.py
or, to specify additional arguments and options;
    streamlit run [OPTIONS] app/app.py [ARGS]
it will occupy the terminal until you close the window
    it spins up a local streamlit server that you can go to
    on Windows, it has an issue releasing the terminal
    you have to do Ctrl + C in the terminal (which won't work)
    then open the page again using the URL that it writes in the terminal
basic concepts:
    https://docs.streamlit.io/get-started/fundamentals/main-concepts
Tako's app:
    https://multioutreg-report.streamlit.app/
Example CSV datasets for his app:
    https://github.com/takotime808/multioutreg/tree/streamlit_app/docs/_static/example_datasets
Tako's repo file that creates it:
    https://github.com/takotime808/multioutreg/blob/streamlit_app/multioutreg/gui/Grid_Search_Surrogate_Models.py
Tako says that he has a way to host it for free
"""

import streamlit as st
import pandas as pd

# st.write is the swiss army knife method for adding stuff to streamlit app
st.write("Streamlit can display a wide variety of data easily:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))