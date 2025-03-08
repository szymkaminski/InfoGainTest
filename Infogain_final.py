import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# from openai import OpenAI
# import openai as openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
import streamlit as st
import sqlite3
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

### FUNCTION DEFINITIONS
# funkcja która zwraca tekst w tabeli markdown
def run_query(sql_query):
  con = sqlite3.connect("patients.db")
  try:
    response = pd.read_sql_query(sql_query, con)
    return response.to_markdown()
  except Exception as e:
    return str(e)
  finally:
    con.close()


# downloading data
ds1=pd.read_excel(r"dataset1.xlsm")
ds2=pd.read_excel(r"dataset2.xlsm")

con = sqlite3.connect("patients.db")
ds1.to_sql("healthinfo", con, index=False, if_exists="replace")
ds2.to_sql("physicalactivity", con, index=False, if_exists="replace")


testquery1="select age from healthinfo"
age_info=pd.read_sql_query(testquery1, con)

### DEFINITIONS OF PROMPTS
# description of database
DB_DESCRIPTION = """The database contains data for the patients of Hospital1.


* "healthinfo" table:
Columns:
patient_number - a unique identifier of the patient
blood_pressure_abnormality - indicates whether the patient has blood pressure abnormality. 1 means abnormality, 0 means no abnormality
Level_of_Hemoglobin - level of hemoglobin measured in g/dl
Genetic_Pedigree_Coefficient - displays the genetic pedigree coefficient
Age - shows the age of the patient
BMI - displays the BMI of the patient
Sex - sex of the patient, 0 for male and 1 for female
Pregnancy - indicates whether the patient is pregnant. 1 means pregnant, 0 means not pregnant
smoking - indicates whether the patient smokes. 1 means smokes, 0 means does not smoke
salt_content_in_the_diet - indicates the salt content in the diet, measured in miligrams per day
alcohol_consumption_per_day - indicates the alcohol consumption per day, measured in mililitres per day
Level_of_Stress - measures the level of stress (cortisol secretion), where 1 means low, 2 means normal and 3 means high
Chronic_kidney_disease - indicates whether the patient has a chronic kidney disease, 0 means no and 1 means yes
Adrenal_and_thyroid_disorders - indicates whether the patient has adrenal and thyroid disorders, 0 means no and 1 means yes


* "physicalactivity" table:
Columns:
patient_number - a unique identifier of the patient, links to the "healthinfo" table
Day_number - number of day where the number of steps was measured
physical_activity - the number of steps taken per day
"""

# check if the chatbot is able to provide an answer to the question asked by the user
can_answer_prompt = PromptTemplate(
    template="""You are a database reading bot that can answer user's question using information from the database.

    Database description:
    """ + DB_DESCRIPTION + """

    You remember all previous interactions with the user. 
    Conversation history:
    {history}

    Given the user's question, decide whether the question can be answered using the information from the database.

    Below are some examples of questions that the user may ask:
    Question: Find the number of people who have a chronic kidney disease.
    Answer: I can find the number of people who have a chronic kidney disease by counting the number of 1 values in the Chronic_kidney_disease column of the healthinfo table
    Question: Count the number of patients with high level of stress and walking on average more than 10000 steps per day.
    Answer "I can find the number of patients with high level of stress and walking on average more than 10000 steps per day by joining the healthinfo table with the physicalactivity table and, filtering by Level_of_Stress=3 and grouping by patient_number and averaging the column physical_activity in table physicalactivity"
    Question: Count the number of people who practice some sports.
    Answer: "The dataset does not contain info whether patients are practicing sports."

    IF you can answer a question, generate a SQL query to retrieve the data required to answer the user's question.Return the SQL query with no explanation and no markdown characters.
    If the user asks for an histogram, please return the data necessary to obtain the relevant data. Examples:
    Question: Please provide an histogram for the age of patients
    Answer: select age from healthinfo
    
    Question: {input}
    """,
    input_variables=["history", "input"]
)

final_answer_prompt = PromptTemplate(
    template="""You are a database reading bot that can answer user's question using information from the database.

    You remember all previous interactions with the user. 
    Conversation history:
    {history}

    Database description:
    """ + DB_DESCRIPTION + """

    Based on the result of the sql query please provide to the user a descriptive answer to his question.
    

    Question: {input}
    """,
    input_variables=["history", "input"]
)

# Page configuration
st.set_page_config(page_title="Szymon Kaminski's Chatbot", page_icon="💬")
st.title("Szymon Kaminski's Chatbot")

# Create a sidebar for API key input
with st.sidebar:
    st.header("API Key Configuration")
    GROQ_API_KEY = st.text_input("Enter your Groq API Key:", type="password")
    api_key_submitted = st.button("Submit API Key")

# Initialize session state for API key status
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

# Update API key status when submitted
if api_key_submitted and GROQ_API_KEY:
    st.session_state.api_key_set = True
    st.session_state.groq_api_key = GROQ_API_KEY
    st.sidebar.success("API Key successfully set!")


# Initialize session state to store conversation history and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    # Use consistent naming - stick with "input" throughout
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        input_key="input",      # Use "input" as the key
        memory_key="history"    # Use "history" as expected by testprompt
    )

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# OpenAI API key input 
# API_KEY = ""
# GROQ_API_KEY=''

# Initialize the OpenAI model
# model=ChatOpenAI(openai_api_key=API_KEY, model="gpt-4o", streaming=True)




if not st.session_state.api_key_set:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
else:
    try:
        model = ChatGroq(groq_api_key=st.session_state.groq_api_key,model_name="llama-3.3-70b-versatile", streaming=True)

        # Create a ConversationChain with memory and prompt
        conversation = ConversationChain(
            llm=model,
            memory=st.session_state.memory,
            prompt=can_answer_prompt,
            verbose=True
        )

        conversation2 = ConversationChain(
            llm=model,
            memory=st.session_state.memory,
            prompt=final_answer_prompt,
            verbose=True
        )


        # Display the chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input using a single chat input field
        user_question = st.chat_input("Ask a question...")

        if user_question:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Display user message in chat container
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Display assistant response in chat container
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = conversation.predict(
                    input=user_question,    # Use "input" as the key
                    callbacks=[st_callback]
                )
                finalanswer=run_query(response)
                
                # st.markdown(finalanswer)

                final_response = conversation2.predict(
                    input=finalanswer,    # Use "input" as the key
                    callbacks=[st_callback]
                )
                
                st.markdown(final_response)
                # GenerateHistogram(finalanswer["Age"],"Age")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.messages.append({"role": "assistant", "content": final_response})
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if "groq_api_key" in st.session_state:
            st.warning("Please check if your API key is valid and try again.")

