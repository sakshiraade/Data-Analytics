import os
import streamlit as st
import openai
import boto3
import json
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Load sensitive information from environment variables
openai.api_key = st.secrets['OPENAI_API_KEY']

# AWS S3 Configuration
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.secrets['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=st.secrets['AWS_SECRET_ACCESS_KEY'],
    region_name=st.secrets['AWS_REGION']
)
s3_bucket = 'gaiaproject'

# Load metadata.jsonl from S3
def load_metadata():
    metadata_key = 'gaia/2023/validation/metadata.jsonl'  # Path in your S3 bucket
    response = s3_client.get_object(Bucket=s3_bucket, Key=metadata_key)
    content = response['Body'].read().decode('utf-8')
    metadata_lines = content.splitlines()
    metadata = [json.loads(line) for line in metadata_lines]
    return pd.DataFrame(metadata)

# Function to load the file from S3 and return its content
def load_file_from_s3(file_name):
    try:
        file_key = f"gaia/2023/validation/{file_name}"  
        response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        return file_content
    except Exception as e:
        return f"Failed to load file: {e}"

# Moderation function
def check_moderation(text):
    moderation_response = openai.Moderation.create(input=text)
    return moderation_response['results'][0]['flagged']

# Send a prompt to the OpenAI model using gpt-3.5-turbo
def query_openai_model(question, context, file_content=None):
    prompt = f"Context: {context}\n\nQuestion: {question}"
    
    # If there is an attached file, include its content in the prompt
    if file_content:
        prompt += f"\n\nAttached File Content: {file_content}\nAnswer:"
    else:
        prompt += "\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response['choices'][0]['message']['content'].strip()

# Compare OpenAI's answer to the final answer in the metadata
def compare_answers(openai_answer, final_answer):
    # Check if the final answer is contained within the OpenAI answer (case-insensitive)
    return final_answer.lower() in openai_answer.lower()

# Option to modify Annotator steps with moderation
def moderate_and_query_openai(question, context, modified_steps, file_content):
    # Check if the modified steps pass moderation
    if check_moderation(modified_steps):
        return "The modified steps violate the content policy. Please revise."
    else:
        # Proceed with re-evaluation if the steps pass moderation
        return query_openai_model(question, modified_steps, file_content)

# Initialize records for test cases (in-memory for simplicity)
if "records" not in st.session_state:
    st.session_state["records"] = {}

if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}

# Track the revised OpenAI answer and the current test case
if "revised_openai_answer" not in st.session_state:
    st.session_state["revised_openai_answer"] = ""  # Store the revised OpenAI answer

if "openai_answer" not in st.session_state:
    st.session_state["openai_answer"] = ""  # Store the OpenAI answer for initial query

if "current_test_case" not in st.session_state:
    st.session_state["current_test_case"] = ""  # Track current test case

# Main Streamlit app
def main():
    st.title("GAIA Dataset Model Evaluation Tool - Team 8")

    # Load and display metadata from S3
    st.header("Validation Test Case Selection")
    metadata_df = load_metadata()
    test_case_options = [f"{i+1}. {task_id}" for i, task_id in enumerate(metadata_df["task_id"].unique())]

    col1, col2 = st.columns([2, 1])

    selected_test_case_option = col1.selectbox("**Select a Test Case**", test_case_options)

    # Assigning id
    test_case_id = selected_test_case_option.split(". ", 1)[1]

    # Reset answers when changing test case
    if test_case_id != st.session_state["current_test_case"]:
        st.session_state["current_test_case"] = test_case_id
        st.session_state["revised_openai_answer"] = ""
        st.session_state["openai_answer"] = ""

    selected_test_case = metadata_df[metadata_df["task_id"] == test_case_id].iloc[0]
    st.write("**Question:**", selected_test_case["Question"])

    # Display attached file name (if applicable)
    file_name = selected_test_case.get("file_name", "")
    if file_name:
        col2.write(f"Attached File: {file_name}")
        file_content = load_file_from_s3(file_name)  # Load the file content from S3
    else:
        file_content = None

    # Display the expected final answer
    final_answer = selected_test_case["Final answer"]
    st.write("**Expected Final Answer:**", final_answer)

    # Query OpenAI with the selected test case
    if st.button("Ask OpenAI"):
        openai_answer = query_openai_model(selected_test_case["Question"], selected_test_case.get("Annotator Metadata", {}).get("Steps", ""), file_content)
        st.session_state["openai_answer"] = openai_answer  # Store OpenAI answer in session state

    # Display OpenAI's answer if asked
    if st.session_state["openai_answer"]:
        st.write("**OpenAI Answer:**", st.session_state["openai_answer"])

        # Compare OpenAI answer with final answer
        correct = compare_answers(st.session_state["openai_answer"], final_answer)
        st.write(f"**Is OpenAI's answer correct?** {'Yes' if correct else 'No'}")

        # Option to modify Annotator steps
        if not correct:
            st.write("Modify the Annotator Steps to improve the model:")
            modified_steps = st.text_area("Annotator Steps", selected_test_case["Annotator Metadata"]["Steps"], key="modified_steps")

            # Re-evaluation button logic with moderation check
            if st.button("Re-evaluate"):
                moderated_response = moderate_and_query_openai(selected_test_case["Question"], selected_test_case.get("Annotator Metadata", {}).get("Steps", ""), modified_steps, file_content)
                st.session_state["revised_openai_answer"] = moderated_response

    # Always display the revised OpenAI answer if it exists for the selected test case
    if st.session_state["revised_openai_answer"]:
        st.write("**Revised OpenAI Answer:**", st.session_state["revised_openai_answer"])

        # Compare the revised answer with the final answer
        if compare_answers(st.session_state["revised_openai_answer"], final_answer):
            st.write("The revised OpenAI answer is correct.")
        else:
            st.write("The revised OpenAI answer is still incorrect.")

    # Buttons for manually assigning the test case
    st.write("**Assign the test case after reviewing the OpenAI response:**")
    col1, col2, col3 = st.columns(3)  # Added third column for "Inconclusive"

    with col1:
        if st.button("Assign As is"):
            st.session_state["records"][test_case_id] = "As is"
            st.write(f"Test case {test_case_id} marked as 'As is'.")

    with col2:
        if st.button("Assign With steps"):
            st.session_state["records"][test_case_id] = "With steps"
            st.write(f"Test case {test_case_id} marked as 'With steps'.")

    with col3:
        if st.button("Assign Inconclusive"):
            st.session_state["records"][test_case_id] = "Inconclusive"
            st.write(f"Test case {test_case_id} marked as 'Inconclusive'.")

    # Feedback section
    st.header("User Feedback")
    feedback = st.text_area("**Provide your feedback on this evaluation**")
    if st.button("Submit Feedback"):
        st.session_state["feedback"][test_case_id] = feedback
        st.write("Thank you for your feedback!")

    # Display a table of all test cases and their statuses
    st.header("Test Case Summary Table")
    table_data = []

    # Loop through metadata and also assign serial numbers to test cases
    for idx, test_case in enumerate(metadata_df["task_id"].unique(), start=1): 
        row = {
            "Test Case": test_case,
            "Prompt": metadata_df[metadata_df["task_id"] == test_case]["Question"].values[0],
            "File Attached": "Yes" if metadata_df[metadata_df["task_id"] == test_case]["file_name"].values[0] else "No",
            "Level": metadata_df[metadata_df["task_id"] == test_case]["Level"].values[0], 
            "Status": st.session_state["records"].get(test_case, "Untested"),
            "User Feedback": st.session_state["feedback"].get(test_case, "N/A")
        }
        table_data.append(row)

    table_df = pd.DataFrame(table_data)
    table_df.index = range(1, len(table_df) + 1) # Set table starting index to 1
    st.dataframe(table_df)

    # Visualize the records using a histogram
    st.header("Records Summary")
    
    # Get counts of each category
    record_df = pd.DataFrame.from_dict(st.session_state["records"], orient='index', columns=["Status"])
    summary = record_df["Status"].value_counts().reindex(["As is", "With steps", "Inconclusive"], fill_value=0)

    # Plot the histogram
    st.write("**Histogram of test cases:**")
    # Check if there is any data to display before plotting
    if summary.sum() > 0:  # Ensure there is data in the summary
        fig, ax = plt.subplots()
        summary.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
        ax.set_xlabel("Test Case Status")
        ax.set_ylabel("Count")
        ax.set_title("Summary of Test Case Status (As is, With steps, Inconclusive)")
        st.pyplot(fig)
    else:
        st.write("No data available to display histogram.")

    # Pie chart for test case statuses
    st.write("**Pie Chart of Test Case Statuses:**")

    summary = summary.fillna(0)  # Fill NaN with 0
    summary = summary.astype(int)  # Convert to integers

    # Check if there are any valid values to plot
    if summary.sum() > 0:
        # Create the pie chart
        fig, ax = plt.subplots()
        ax.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90, colors=['green', 'orange', 'red'])
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is a circle
        st.pyplot(fig)
    else:
        st.write("No data available to display the pie chart.")

    # Bar chart showing how many test cases of each level have what status
    st.write("**Bar Chart of Test Case Statuses and Level:**")

    # Group data by Level and Status, then count occurrences
    level_status_df = pd.merge(metadata_df[['task_id', 'Level']], record_df, left_on='task_id', right_index=True)

    # Group by Level and Status, then count occurrences, making sure the counts are numeric
    level_status_summary = level_status_df.groupby(['Level', 'Status']).size().unstack(fill_value=0)

    # Check if there is any data to display before plotting
    if not level_status_summary.empty:
        fig, ax = plt.subplots()
        level_status_summary.plot(kind='bar', stacked=True, ax=ax, color=['green', 'orange', 'red'])
        ax.set_xlabel("Test Case Level")
        ax.set_ylabel("Count")
        ax.set_title("Test Case Status by Level")
        st.pyplot(fig)
    else:
        st.write("No data available to display bar chart.")

if __name__ == "__main__":
    main()
