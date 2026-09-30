"""Lightweight inspection-results viewer.

Run after analysing a video:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RoadVision AI", layout="wide")
st.title("RoadVision AI")
st.caption("Road damage detection, depth-assisted severity scoring and GPS inspection records")

csv_file = st.file_uploader("Upload a RoadVision events CSV", type=["csv"])
if csv_file is None:
    st.info("Analyse a road video first, then upload the generated events.csv file.")
    st.stop()

df = pd.read_csv(csv_file)

c1, c2, c3 = st.columns(3)
c1.metric("Unique defects", len(df))
c2.metric("High severity", int((df.get("severity_label") == "high").sum()))
c3.metric("Average severity", f"{df.get('severity_score', pd.Series(dtype=float)).mean():.1f}")

st.subheader("Inspection records")
st.dataframe(df, use_container_width=True)

if {"latitude", "longitude"}.issubset(df.columns):
    map_df = df[["latitude", "longitude"]].dropna().rename(columns={"latitude": "lat", "longitude": "lon"})
    if not map_df.empty:
        st.subheader("Defect map")
        st.map(map_df)
