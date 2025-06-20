import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit UI Configuration
st.set_page_config(
    page_icon="🗄️",
    page_title="Chat with MySQL Database",
    layout="centered"
)
def connectDatabase(username, port, host, password, database):
    """Connect to MySQL database with error handling"""
    try:
        mysql_uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
        st.session_state.db = SQLDatabase.from_uri(mysql_uri)
        return True, "Database connected successfully"
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False, f"Database connection failed: {str(e)}"

def runQuery(query):
    """Execute SQL query with error handling"""
    if not hasattr(st.session_state, 'db') or st.session_state.db is None:
        return "Please connect to database first"
    
    try:
        # Clean the query
        clean_query = cleanSQLQuery(query)
        logger.info(f"Executing query: {clean_query}")
        result = st.session_state.db.run(clean_query)
        return result
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return f"Query execution failed: {str(e)}"

def getDatabaseSchema():
    """Get database schema with error handling"""
    if not hasattr(st.session_state, 'db') or st.session_state.db is None:
        return "Please connect to database first"
    
    try:
        return st.session_state.db.get_table_info()
    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        return f"Failed to get schema: {str(e)}"

def cleanSQLQuery(query):
    """Clean and validate SQL query"""
    # Remove any markdown formatting
    query = re.sub(r'```sql\s*', '', query)
    query = re.sub(r'```\s*', '', query)
    
    # Remove extra whitespace and newlines
    query = ' '.join(query.split())
    
    # Ensure query ends with semicolon if it doesn't have one
    if not query.strip().endswith(';'):
        query = query.strip() + ';'
    
    return query

# Initialize LLM with CUDA support
@st.cache_resource
def initialize_llm():
    """Initialize ChatOllama with CUDA support"""
    try:
        # Configure for CUDA support
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.1,  # Lower temperature for more consistent SQL generation
            num_gpu=1,  # Use 1 GPU
            num_thread=8,  # Adjust based on your CPU cores
            # Additional parameters for better performance
            top_k=10,
            top_p=0.9,
            repeat_penalty=1.1
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM with CUDA: {e}")
        # Fallback to CPU
        return ChatOllama(model="llama3.2", temperature=0.1)

llm = initialize_llm()

def getQueryFromLLM(question):
    """Generate SQL query from natural language question with improved prompting"""
    template = """You are a SQL expert. Given the database schema below, generate ONLY a valid SQL query to answer the user's question.

DATABASE SCHEMA:
{schema}

IMPORTANT RULES:
1. Return ONLY the SQL query, nothing else
2. Use proper table and column names from the schema (case-sensitive)
3. Always use single quotes for string values
4. End the query with a semicolon
5. Use proper SQL syntax for the database type (MySQL)
6. For COUNT queries, use COUNT(*) or COUNT(column_name)
7. For string comparisons, use proper quoting and consider case sensitivity

EXAMPLES:
Question: How many albums are in the database?
SQL: SELECT COUNT(*) FROM album;

Question: How many customers are from Brazil?
SQL: SELECT COUNT(*) FROM customer WHERE country = 'Brazil';

Question: List all customers from India
SQL: SELECT * FROM customer WHERE country = 'India';

Question: What are the top 5 albums by name?
SQL: SELECT * FROM album ORDER BY title LIMIT 5;

Now generate the SQL query for this question:
Question: {question}
SQL:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": question,
            "schema": getDatabaseSchema()
        })
        
        # Extract just the SQL query from the response
        sql_query = response.content.strip()
        
        # Additional cleaning
        sql_query = cleanSQLQuery(sql_query)
        
        logger.info(f"Generated SQL query: {sql_query}")
        return sql_query
        
    except Exception as e:
        logger.error(f"Failed to generate SQL query: {e}")
        return f"Error generating query: {str(e)}"

def getResponseForQueryResult(question, query, result):
    """Generate natural language response from query results"""
    template = """You are a helpful database assistant. Based on the user's question, SQL query, and results, provide a clear, natural language response.

DATABASE SCHEMA:
{schema}

CONVERSATION CONTEXT:
Question: {question}
SQL Query: {query}
Query Result: {result}

INSTRUCTIONS:
1. Provide a clear, conversational response
2. Include the actual numbers/data from the result
3. Be helpful and informative
4. If there's an error in the result, explain it clearly
5. Keep the response concise but complete

EXAMPLES:
Question: How many albums are in the database?
Result: [(34,)]
Response: There are 34 albums in the database.

Question: List customers from Brazil
Result: [('João Silva', 'Brazil'), ('Maria Santos', 'Brazil')]
Response: I found 2 customers from Brazil: João Silva and Maria Santos.

Question: What is the total sales?
Result: [(15000.50,)]
Response: The total sales amount is $15,000.50.

Now provide your response:
Response:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": question,
            "schema": getDatabaseSchema(),
            "query": query,
            "result": str(result)
        })
        
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return f"I encountered an error while processing the results: {str(e)}"



st.title("CA ML Chat with MySQL Database")
# st.markdown("Ask questions about your database in natural language!")

# Initialize session state
if "chat" not in st.session_state:
    st.session_state.chat = []

if "db_connected" not in st.session_state:
    st.session_state.db_connected = False

# Main chat interface
question = st.chat_input('Ask a question about your database...')

if question:
    if not st.session_state.db_connected:
        st.error('⚠️ Please connect to the database first using the sidebar.')
    else:
        # Add user message to chat
        st.session_state.chat.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Generating SQL query..."):
            # Generate SQL query
            query = getQueryFromLLM(question)
            
        with st.spinner("Executing query..."):
            # Execute query
            result = runQuery(query)
            
        with st.spinner("Generating response..."):
            # Generate natural language response
            response = getResponseForQueryResult(question, query, result)
            
        # Add assistant response to chat
        st.session_state.chat.append({
            "role": "assistant",
            "content": response,
            "query": query,
            "result": str(result)
        })

# Display chat history
for i, chat in enumerate(st.session_state.chat):
    with st.chat_message(chat['role']):
        st.markdown(chat['content'])
        
        # Show SQL query and result for assistant messages
        if chat['role'] == 'assistant' and 'query' in chat:
            with st.expander("View SQL Query & Results"):
                st.code(chat['query'], language='sql')
                st.text("Result:")
                st.code(chat['result'])

# Sidebar for database connection
with st.sidebar:
    st.title('🔗 Database Connection')
    
    # Connection status
    if st.session_state.db_connected:
        st.success("✅ Database Connected")
    else:
        st.warning("⚠️ Not Connected")
    
    st.markdown("---")
    
    # Connection parameters
    host = st.text_input(label="Host", value="localhost")
    port = st.text_input(label="Port", value="3306")
    username = st.text_input(label="Username", value="root")
    password = st.text_input(label="Password", type="password")
    database = st.text_input(label="Database Name")
    
    connect_btn = st.button("🔌 Connect to Database", type="primary")
    
    if connect_btn:
        if not all([host, port, username, password, database]):
            st.error("Please fill in all connection details.")
        else:
            with st.spinner("Connecting to database..."):
                success, message = connectDatabase(username, port, host, password, database)
                
            if success:
                st.session_state.db_connected = True
                st.success(message)
                st.rerun()
            else:
                st.session_state.db_connected = False
                st.error(message)
    
    # Disconnect button
    if st.session_state.db_connected:
        if st.button("🔌 Disconnect"):
            if hasattr(st.session_state, 'db'):
                del st.session_state.db
            st.session_state.db_connected = False
            st.success("Disconnected from database")
            st.rerun()
    
    st.markdown("---")
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat = []
        st.success("Chat history cleared")
        st.rerun()
    
    # Show database schema
    if st.session_state.db_connected:
        with st.expander("📋 Database Schema"):
            schema = getDatabaseSchema()
            st.text(schema)

# Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center; color: #666;'>
#         Built with Streamlit • Powered by Ollama with CUDA
#     </div>
#     """, 
#     unsafe_allow_html=True
# )