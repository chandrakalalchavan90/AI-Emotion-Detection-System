import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# PAGE TITLE
st.title("AI Emotion Analytics Dashboard")

# LOAD CSV
df = pd.read_csv("emotion_log.csv")

# SHOW DATA
st.subheader("Emotion Detection Data")
st.dataframe(df)

# TOTAL DETECTIONS
st.subheader("Total Detections")
st.success(len(df))

# LATEST EMOTION
st.subheader("Latest Detected Emotion")
st.info(df.iloc[-1]["Emotion"])

# DOWNLOAD BUTTON
with open("emotion_log.csv", "rb") as file:
    st.download_button(
        label="Download Emotion Report",
        data=file,
        file_name="emotion_log.csv",
        mime="text/csv"
    )

# EMOTION COUNT
emotion_count = df["Emotion"].value_counts()

# PIE CHART
st.subheader("Emotion Distribution")

fig1, ax1 = plt.subplots()

ax1.pie(
    emotion_count,
    labels=emotion_count.index,
    autopct='%1.1f%%'
)

st.pyplot(fig1)

# BAR GRAPH
st.subheader("Emotion Frequency")

fig2, ax2 = plt.subplots()

ax2.bar(
    emotion_count.index,
    emotion_count.values
)

ax2.set_xlabel("Emotion")
ax2.set_ylabel("Count")

st.pyplot(fig2)

# MOOD HISTORY GRAPH
st.subheader("Mood History")

fig3, ax3 = plt.subplots()

ax3.plot(df.index, df["Confidence"])

ax3.set_xlabel("Detections")
ax3.set_ylabel("Confidence")

st.pyplot(fig3)