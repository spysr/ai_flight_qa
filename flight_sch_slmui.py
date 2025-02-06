############### STREAMLIT UI used for UI ###########
############### Transformer Used all-MiniLM-L6-v2 ###########

import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the flight schedule CSV
file_path = "ivtest-sched.csv"  # Update this with the correct path if needed
df = pd.read_csv(file_path, names=["FlightNumber", "Origin", "Destination", "DepartureTime", "ArrivalTime"])

# Check for missing or empty values
df["DepartureTime"] = df["DepartureTime"].astype(str).str.strip().replace("", "0000").fillna("0000")
df["ArrivalTime"] = df["ArrivalTime"].astype(str).str.strip().replace("", "0000").fillna("0000")

# Ensure all values are numeric before conversion
df = df[df["DepartureTime"].str.isnumeric()]
df = df[df["ArrivalTime"].str.isnumeric()]

df["DepartureTime"] = df["DepartureTime"].str.zfill(4)
df["ArrivalTime"] = df["ArrivalTime"].str.zfill(4)

# Convert HHMM format to HH:MM
df["DepartureTime"] = df["DepartureTime"].apply(lambda x: f"{x[:2]}:{x[2:]}")
df["ArrivalTime"] = df["ArrivalTime"].apply(lambda x: f"{x[:2]}:{x[2:]}")

# Calculate Flight Duration
df["DepartureMinutes"] = df["DepartureTime"].apply(lambda x: int(x[:2]) * 60 + int(x[3:]))
df["ArrivalMinutes"] = df["ArrivalTime"].apply(lambda x: int(x[:2]) * 60 + int(x[3:]))
df["FlightDuration"] = df["ArrivalMinutes"] - df["DepartureMinutes"]

# Define time categories for flights
def categorize_flight_time(dep_time):
    hour = int(dep_time[:2])
    if 5 <= hour < 12:
        return "Morning Flight"
    elif 12 <= hour < 17:
        return "Afternoon Flight"
    elif 17 <= hour < 21:
        return "Evening Flight"
    else:
        return "Night Flight"

df["FlightCategory"] = df["DepartureTime"].apply(categorize_flight_time)

# Generate Comprehensive Q&A Pairs
qa_pairs = []

for _, row in df.iterrows():
    flight_num = row["FlightNumber"]
    origin = row["Origin"]
    destination = row["Destination"]
    dep_time = row["DepartureTime"]
    arr_time = row["ArrivalTime"]
    duration = row["FlightDuration"]
    category = row["FlightCategory"]

    # General Information
    qa_pairs.append((f"What is the flight number for the route from {origin} to {destination}?", f"The flight number for the route from {origin} to {destination} is {flight_num}."))
    qa_pairs.append((f"What are the scheduled departure and arrival times for flight {flight_num}?", f"Flight {flight_num} departs at {dep_time} and arrives at {arr_time}."))
    qa_pairs.append((f"What is the duration of flight {flight_num}?", f"Flight {flight_num} takes approximately {duration // 60} hours and {duration % 60} minutes."))

    # Time-Based Queries
    qa_pairs.append((f"What are the {category.lower()} flights from {origin} to {destination}?", f"Flight {flight_num} from {origin} to {destination} is a {category}, departing at {dep_time}."))

# Convert to a DataFrame
qa_df = pd.DataFrame(qa_pairs, columns=["Question", "Answer"])

# Load Pretrained SentenceTransformer Model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Encode questions and create FAISS index
@st.cache_data
def create_faiss_index():
    question_embeddings = model.encode(qa_df["Question"].tolist())
    index = faiss.IndexFlatL2(question_embeddings.shape[1])
    index.add(np.array(question_embeddings))
    return index

index = create_faiss_index()

# Define Function to Retrieve the Best Answer
def get_answer(query):
    query_embedding = model.encode([query])
    _, idx = index.search(np.array(query_embedding), 1)
    return qa_df.iloc[idx[0][0]]["Answer"]

#Step 5: Streamlit UI
st.title("Flight Schedule Q&A System ✈️")
st.write("Ask any flight-related question based on the schedule!")

user_query = st.text_input("Enter your question:")

if user_query:
    response = get_answer(user_query)
    st.success(response)
