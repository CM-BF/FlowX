import streamlit as st
import pandas as pd
import copy
import pyarrow as pa
from streamlit.elements.data_editor import _apply_dataframe_edits, determine_dataframe_schema
import numpy as np

st.title("Job Launcher")


# if 'gpu_monitor' not in st.session_state:
#
#     data = pd.DataFrame([[False, 'Empty'] for i in range(10)], columns=['GPU', 'Jobs'], index=[i for i in range(10)])
#     st.session_state.gpu_monitor = data
@st.cache_resource
def load_gpu_monitor():
    data = pd.DataFrame([[False, 'Empty'] for i in range(10)], columns=['GPU', 'Jobs'], index=[i for i in range(10)])
    return data

# data = pd.DataFrame([[False, 'Empty'] for i in range(10)], columns=['GPU', 'Jobs'], index=[i for i in range(10)])
gpu_monitor = load_gpu_monitor()
# st.write(id(gpu_monitor))
if 'orig_monitor' not in st.session_state:
    st.session_state.orig_monitor = copy.deepcopy(gpu_monitor)
edited_monitor = st.data_editor(st.session_state.orig_monitor, key='edited_monitor')

# def edit_gpu_monitor(gpu_monitor, edited_rows):
#     for idx in edited_rows.keys():
#         for col, value in edited_rows[idx].items():
#             gpu_monitor.loc[idx, col] = value
#     # return gpu_monitor
# # gpu_monitor = edited_monitor
# # st.write(id(gpu_monitor))
#
# edit_gpu_monitor(gpu_monitor, st.session_state.edited_monitor["edited_rows"])

def edit_gpu_monitor(gpu_monitor):
    arrow_table = pa.Table.from_pandas(gpu_monitor)

    # Determine the dataframe schema which is required for parsing edited values
    # and for checking type compatibilities.
    dataframe_schema = determine_dataframe_schema(gpu_monitor, arrow_table.schema)
    _apply_dataframe_edits(gpu_monitor, st.session_state.edited_monitor, dataframe_schema)

edit_gpu_monitor(gpu_monitor)

st.markdown(f'You choose: {gpu_monitor["GPU"].to_numpy().nonzero()[0]}')

# gpu_monitor
# st.session_state.gpu_monitor = gpu_monitor
#
# st.write(type(st.session_state.to_dict()["gpu_monitor"]))



# class MockAPI:
#     def get_jobs(self):
#         return ["job1", "job2", "job3"]
#
#     def get_status(self, job):
#         return "running"
#
# api = MockAPI()
#
# if 'jobs' not in st.session_state:
#     st.session_state['jobs'] = api.get_jobs()
#
# def refresh_jobs():
#     st.session_state['jobs'] = api.get_jobs()
#
# def run_jobs():
#     # Code to run jobs using AdaLauncher
#     pass
#
#
#
# if st.button("Refresh Jobs", on_click=refresh_jobs):
#     pass
#
# if st.button("Run Jobs", on_click=run_jobs):
#     pass
#
# for job in st.session_state['jobs']:
#     st.text(f"{job}: {api.get_status(job)}")
