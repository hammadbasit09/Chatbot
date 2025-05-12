import streamlit as st
import pandas as pd
import openai
import os
import matplotlib.pyplot as plt
import io
import base64
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import (ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder, PromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import re
from dotenv import load_dotenv

## Hammad work
# Set page config
st.set_page_config(page_title="Music Store SQL Bot", layout="wide")


# Create custom callback handler for SQL queries
class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = []

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can
         record it as the final sql"""
        if action.tool in ["sql_db_query_checker", "sql_db_query"]:
            self.sql_result.append(action.tool_input)


# Function to remove Python code blocks
def remove_python_code(response_text):
    """Removes Python code blocks and surrounding explanatory text."""
    # Remove the entire visualization suggestion along with the code block
    cleaned_text = re.sub(r"To help visualize this data,.*?This code will generate.*?", "", response_text,
                          flags=re.DOTALL)

    # Strip extra spaces
    return cleaned_text.strip()


# Initialize session state for chat history and visualization if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_visualization' not in st.session_state:
    st.session_state.last_visualization = None


# Function to remove Python code blocks
def remove_python_code(response_text):
    """Removes Python code blocks enclosed in triple backticks from the chatbot output."""
    cleaned_text = re.sub(r"```python.*?```", "", response_text, flags=re.DOTALL)
    return cleaned_text.strip()


# Replace with your actual OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Function to render visualization from Python code
def render_visualization(code):
    try:
        # Create a buffer to capture the plot
        buffer = io.BytesIO()

        # Execute the visualization code
        local_vars = {}
        exec(code, globals(), local_vars)

        # Save the plot to the buffer
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode the image
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()  # Close the plot to free up memory

        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


# Function to initialize the agent
@st.cache_resource
def initialize_agent():
    try:
        # Set OpenAI API key
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key  # Set environment variable as backup

        # Connect to the database
        # IMPORTANT: Replace with the correct path to your database
        db = SQLDatabase.from_uri(
            f"sqlite:///C:/Users/Hammad/Downloads/Chinook.sqlite")

        # Initialize the LLM
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

        # Example queries for few-shot learning
        examples = [
            {"input": "Calculate the total revenue generated from all invoices in 2024.",
             "query": "SELECT SUM(Total) AS TotalRevenue FROM Invoice WHERE STRFTIME('%Y', InvoiceDate) = '2024';"},
            {"input": "Show the average invoice value for each customer along with their names.",
             "query": "SELECT c.CustomerId, c.FirstName, c.LastName, AVG(i.Total) AS AverageInvoiceValue FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId;"},
            {"input": "List customers who have spent more than $100 in total. Include their names and total amount spent.",
             "query": "SELECT c.CustomerId, c.FirstName, c.LastName, SUM(i.Total) AS TotalSpent FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId HAVING SUM(i.Total) > 100;"},
            {"input": "Show monthly revenue trend for the year 2024.",
             "query": "SELECT STRFTIME('%Y-%m', InvoiceDate) AS Month, SUM(Total) AS MonthlyRevenue FROM Invoice WHERE STRFTIME('%Y', InvoiceDate) = '2024' GROUP BY Month ORDER BY Month;"},
            {"input": "List the top 5 most expensive tracks by unit price.",
             "query": "SELECT TrackId, Name, UnitPrice FROM Track ORDER BY UnitPrice DESC LIMIT 5;"},
            {"input": "Calculate total sales generated by each employee. Show employee names and total sales.",
             "query": "SELECT e.EmployeeId, e.FirstName, e.LastName, SUM(i.Total) AS TotalSales FROM Employee e JOIN Customer c ON e.EmployeeId = c.SupportRepId JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY e.EmployeeId;"},
            {"input": "What is the average price of tracks in each genre?",
             "query": "SELECT g.Name AS Genre, AVG(t.UnitPrice) AS AveragePrice FROM Genre g JOIN Track t ON g.GenreId = t.GenreId GROUP BY g.Name;"},
            {"input": "List albums that have generated the most revenue.",
             "query": "SELECT al.Title AS AlbumTitle, SUM(il.UnitPrice * il.Quantity) AS AlbumRevenue FROM Album al JOIN Track t ON al.AlbumId = t.AlbumId JOIN InvoiceLine il ON t.TrackId = il.TrackId GROUP BY al.Title ORDER BY AlbumRevenue DESC;"},
            {"input": "Calculate total revenue from each media type.",
             "query": "SELECT mt.Name AS MediaType, SUM(il.UnitPrice * il.Quantity) AS TotalRevenue FROM MediaType mt JOIN Track t ON mt.MediaTypeId = t.MediaTypeId JOIN InvoiceLine il ON t.TrackId = il.TrackId GROUP BY mt.Name;"},
            {"input": "Find the customer who spent the most in total. Show their name and amount spent.",
             "query": "SELECT c.CustomerId, c.FirstName, c.LastName, SUM(i.Total) AS Maximum_Total FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId ORDER BY Maximum_Total DESC LIMIT 1;"}
        ]

        # Create example selector
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(api_key=api_key),
            FAISS,
            k=5,
            input_keys=["input"]
        )

        # System prompts
        system_prefix = """
        You are an agent designed to interact with a SQL database.

        VISUALIZATION INSTRUCTIONS:
        - If a visualization would help explain the data, provide a complete Python code snippet using matplotlib to create a relevant graph.
        - The code should include all necessary imports.
        - Use clear, descriptive labels and titles for the visualization.
        - Ensure the graph provides meaningful insights into the data.

        Given an input question, create a syntactically correct {dialect} query to run, then examine the results and provide the answer.
        Order the results by relevant columns to highlight the most pertinent data.
        You have access to tools for interacting with the database.
        Use only the provided tools and the information they return to construct your final answer.
        """

        system_suffix = """
        After executing the query, present the results to the user in a clear and organized format:
        - Display tabular data with appropriate column headings.
        - Do not truncate or limit the number of rows or columns in the output.
        - Provide a summary of key insights.
        - Generate a matplotlib visualization code snippet.
        - Provide only the visualization code without any introductory text or explanations.
        - The visualization should directly relate to the query results.
        - Provide complete, executable Python code for creating the graph.
        """

        # Create the few-shot prompt
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=PromptTemplate.from_template("User input: {input}\nSQL query: {query}"),
            input_variables=["input", "dialect", "top_k"],
            prefix=system_prefix,
            suffix=system_suffix,
        )

        # Create the full prompt
        full_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # Initialize memory
        memory = ConversationBufferMemory(memory_key='history', input_key='input')

        # Create the SQL agent
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            prompt=full_prompt,
            verbose=True,
            handle_parsing_errors=True,
            agent_type="openai-tools",
            agent_executor_kwargs={'memory': memory}
        )

        return agent_executor, db

    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None, None


# Main Streamlit app
def main():
    # Main app title
    st.title("Music Store Financial Data Analysis")
    st.markdown("Ask questions about sales, revenue, and customer spending.")

    # Chat interface
    user_input = st.text_input("Enter your question about the music store's financial data:",
                               placeholder="e.g., Show the average invoice value for each customer along with their names.")

    # Initialize agent
    with st.spinner("Initializing agent..."):
        agent, db = initialize_agent()

    if agent and user_input:
        with st.spinner("Processing your query..."):
            try:
                # Initialize the SQL handler
                handler = SQLHandler()

                # Execute query through agent with the handler
                response = agent.invoke({"input": user_input}, {"callbacks": [handler]})
                output = response.get('output', 'No response found')
                filtered_output = remove_python_code(output)

                # Get SQL queries from handler
                sql_queries = handler.sql_result

                # For debugging
                print(sql_queries)
                print("Agent:", output)

                # Extract SQL query if present
                sql_query = None
                if sql_queries and len(sql_queries) > 0:
                    # Use the last SQL query executed
                    sql_query = sql_queries[-1]
                elif "```sql" in output:
                    # Fallback to parsing from output
                    sql_start = output.find("```sql")
                    sql_end = output.find("```", sql_start + 6)
                    if sql_start != -1 and sql_end != -1:
                        sql_query = output[sql_start + 6:sql_end].strip()

                # Extract visualization code if present
                viz_code = None
                if "```python" in output:
                    viz_start = output.find("```python")
                    viz_end = output.find("```", viz_start + 9)
                    if viz_start != -1 and viz_end != -1:
                        viz_code = output[viz_start + 9:viz_end].strip()

                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_input,
                    "answer": filtered_output,
                    "sql_query": sql_query,
                    "visualization": None
                })

                # Display the latest response
                st.subheader("Response:")
                st.write(filtered_output)

                # Render Visualization if code is present
                if viz_code:
                    st.subheader("Visualization:")
                    try:
                        # Render and display the visualization
                        viz_image = render_visualization(viz_code)
                        if viz_image:
                            st.image(viz_image, caption="Generated Visualization")

                            # Update the last chat history item with visualization
                            st.session_state.chat_history[-1]['visualization'] = viz_image
                    except Exception as e:
                        st.error(f"Error rendering visualization: {str(e)}")

                # Display SQL Query at the end if found (moved from earlier position)
                if sql_query:
                    st.subheader("SQL Query:")
                    # Only display the query, don't execute it
                    st.code(sql_query, language="sql")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Something went wrong with your query. Please try again.")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            expander = st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['question']}")
            with expander:
                st.markdown(chat['answer'])

                # Display Visualization if available
                if chat['visualization']:
                    st.subheader("Visualization:")
                    st.image(chat['visualization'], caption="Generated Visualization")

                # Display SQL Query at the end if available (within the expander)
                if chat['sql_query']:
                    st.subheader("SQL Query:")
                    st.code(chat['sql_query'], language="sql")

    # Footer
    st.markdown("---")
    st.markdown("Music Store SQL Bot")


# Run the main function
if __name__ == "__main__":
    main()