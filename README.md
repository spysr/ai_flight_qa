Flight Schedule Q&A System

This project provides an AI-powered Flight Schedule Q&A System using LLM (Sentence Transformers) and FAISS (Facebook AI Similarity Search) to answer user queries about flight schedules.

🚀 Features

✅ Interactive UI using Streamlit

✅ Fast and accurate question answering

✅ Supports general, time-based, and airline-specific queries

✅ Uses FAISS for efficient retrieval

✅ Pre-trained LLM (all-MiniLM-L6-v2) for better accuracy

💻 Installation Guide

1️⃣ Install Dependencies

Ensure you have Python 3.8+ installed. Then, install the required libraries:

pip install streamlit faiss-cpu sentence-transformers torch torchvision torchaudio numpy pandas

If you encounter issues with PyTorch, install it explicitly:

For CPU Users:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

For GPU Users (CUDA 11.8):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2️⃣ Run the Application

streamlit run flight_schedule_qa_ui.py

Once the app starts, open your browser and interact with the Q&A system.

🛠️ How It Works

Loads flight schedules from ivtest-sched.csv

Formats Departure & Arrival Times to HH:MM

Calculates flight duration & categories (Morning, Afternoon, Evening, Night)

Generates Q&A pairs based on the schedule

Uses Sentence Transformers to encode questions

Stores them in a FAISS index for fast retrieval

Provides an interactive UI for users to ask queries

🤖 Example Questions to Ask

"What is the fastest flight from DEL to BOM?"

"What are the morning flights from AMD to BLR?"

"What time does flight 459 arrive?"

"What are the most common flight routes?"

"Which airline offers the shortest travel time between two destinations?"

🛠 Troubleshooting

Issue: Streamlit command not found

Try running:

python -m streamlit run flight_schedule_qa_ui.py


📜 License

This project is licensed under GO-MMT

💬 Need Help?

If you have any questions or face issues, feel free to ask! 🚀