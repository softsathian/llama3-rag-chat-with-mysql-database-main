import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit UI Configuration
st.set_page_config(
    page_icon="🗄️",
    page_title="Chat with Database",
    layout="centered"
)

def connectDatabase(username, port, host, password, database):
    """Connect to MySQL database with error handling"""
    try:
        mysql_uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
        st.session_state.db = SQLDatabase.from_uri(mysql_uri)
      
        if "db_connected" in st.session_state:
            st.success("Database connected successfully")
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
    query = re.sub(r'^\s*SQL:\s*', '', query, flags=re.IGNORECASE)
    
    # Remove extra whitespace and newlines
    query = ' '.join(query.split())
    
    # Remove trailing explanatory text after semicolon
    if ';' in query:
        query = query.split(';')[0] + ';'
    
    # Ensure query ends with semicolon if it doesn't have one
    if not query.strip().endswith(';'):
        query = query.strip() + ';'
    
    return query

# Initialize LLM with proper CUDA support
@st.cache_resource
def initialize_llm():
    """Initialize ChatOllama with CUDA support"""
    try:
        # Set environment variable for CUDA (if not already set)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Initialize ChatOllama with correct parameters
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.0,  # Set to 0 for most deterministic results
            base_url="http://localhost:11434",  # Default Ollama URL
        )
        
        # Test the model to ensure it's working
        test_response = llm.invoke("SELECT 1;")
        logger.info("LLM initialized successfully with CUDA support")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        # Try fallback initialization
        try:
            return ChatOllama(model="llama3.2", temperature=0.0)
        except Exception as fallback_error:
            logger.error(f"Fallback initialization also failed: {fallback_error}")
            raise fallback_error

# Check CUDA availability
def check_cuda_status():
    """Check if CUDA is available and being used"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, "NVIDIA GPU detected"
        else:
            return False, "No NVIDIA GPU detected"
    except:
        return False, "nvidia-smi not available"

llm = initialize_llm()

def getQueryFromLLM(question):
    """Generate SQL query from natural language question with security restrictions"""
    template = """You are an expert MySQL database analyst. Your task is to convert natural language questions into precise SQL queries.
DATABASE SCHEMA: {schema}

SECURITY RESTRICTIONS - STRICTLY FORBIDDEN:
- NO DROP statements (DROP TABLE, DROP DATABASE, DROP INDEX, etc.)
- NO DELETE statements (DELETE FROM table_name)
- NO TRUNCATE statements (TRUNCATE TABLE)
- NO ALTER statements (ALTER TABLE, ALTER DATABASE)
- NO CREATE statements (CREATE TABLE, CREATE INDEX)
- NO UPDATE statements (UPDATE table_name SET)
- NO INSERT statements (INSERT INTO)
- ONLY SELECT queries are allowed for data retrieval and analysis

CRITICAL INSTRUCTIONS:
- don't return SELECT queries - no explanations, no markdown, no additional text
- Study the schema carefully - use EXACT table and column names (case-sensitive)
- Use single quotes for string literals: 'value'
- Use backticks for table/column names if they might be MySQL keywords: order
- Always end with semicolon ;
- For aggregations: COUNT(*), SUM(), AVG(), MAX(), MIN()
- For text searches: use LIKE '%pattern%' or LIKE 'pattern%'
- For dates: use proper MySQL date format 'YYYY-MM-DD'
- For joins: use proper JOIN syntax with ON conditions
- For limits: use LIMIT n for top n results
- If asked to modify/delete/create data, respond with: FORBIDDEN_OPERATION

READ-ONLY PATTERNS TO FOLLOW:
- Count records: SELECT COUNT(*) FROM table_name;
- Filter by condition: SELECT * FROM table_name WHERE column = 'value';
- Order results: SELECT * FROM table_name ORDER BY column_name ASC/DESC;
- Top N records: SELECT * FROM table_name ORDER BY column_name LIMIT n;
- Sum/Average: SELECT SUM(column_name) FROM table_name;
- Group by: SELECT column, COUNT(*) FROM table_name GROUP BY column;
- Join tables: SELECT t1.col, t2.col FROM table1 t1 JOIN table2 t2 ON t1.id = t2.foreign_id;
- Use proper JOIN syntax and conditional filters where necessary.
- Search text: SELECT * FROM table_name WHERE column_name LIKE '%search_term%';
- Date range: SELECT * FROM table_name WHERE date_column BETWEEN 'start_date' AND 'end_date';
- Distinct values: SELECT DISTINCT column_name FROM table_name;
- Conditional aggregation: SELECT COUNT(*) FROM table_name WHERE condition;
- Multiple conditions: SELECT * FROM table_name WHERE col1 = 'value' AND col2 > 100;
- Sorting with multiple columns: SELECT * FROM table_name ORDER BY col1 ASC, col2 DESC;
- Grouping with filtering: SELECT column, COUNT() FROM table_name GROUP BY column HAVING COUNT() > 1;

EXAMPLES FOR READ-ONLY OPERATIONS: 
Question: How many records are in the employee table? 
SELECT COUNT(*) FROM employee;

Question: How many albums are in the database?
SQL: SELECT COUNT(*) FROM album;

Question: How many customers are from Brazil?
SQL: SELECT COUNT(*) FROM customer WHERE country = 'Brazil';

Question: List all customers from India
SQL: SELECT * FROM customer WHERE country = 'India';

Question: What are the top 5 albums by name?
SQL: SELECT * FROM album ORDER BY title LIMIT 5;

FORBIDDEN EXAMPLES (DO NOT GENERATE THESE): ❌ Question: Delete all old records ❌ Response: FORBIDDEN_OPERATION
❌ Question: Remove inactive users
❌ Response: FORBIDDEN_OPERATION
❌ Question: Update customer information ❌ Response: FORBIDDEN_OPERATION
❌ Question: Create a new table ❌ Response: FORBIDDEN_OPERATION
❌ Question: Drop the users table ❌ Response: FORBIDDEN_OPERATION
Now convert this question to SQL (ONLY SELECT queries allowed): 

Question: {question}


SQL Query:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": question,
            "schema": getDatabaseSchema()
        })
        
        # Extract SQL query from response
        sql_query = response.content.strip()
        
        # Security check - validate that only SELECT queries are generated
        if not is_safe_query(sql_query):
            logger.warning(f"Unsafe query detected: {sql_query}")
            return "FORBIDDEN_OPERATION: Only SELECT queries are allowed for data retrieval."
        
        # Additional cleaning
        sql_query = cleanSQLQuery(sql_query)
        
        logger.info(f"Generated SQL query: {sql_query}")
        return sql_query
        
    except Exception as e:
        logger.error(f"Failed to generate SQL query: {e}")
        return f"Error generating query: {str(e)}"

def is_safe_query(query):
    """Validate that the query is safe (only SELECT operations)"""
    if not query or query.strip() == "":
        return False
    
    # Convert to uppercase for checking
    query_upper = query.upper().strip()
    
    # Check for forbidden operations
    forbidden_keywords = [
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 
        'UPDATE', 'INSERT', 'REPLACE', 'RENAME', 'GRANT', 
        'REVOKE', 'SET', 'LOCK', 'UNLOCK'
    ]
    
    # Check if query starts with forbidden keywords
    for keyword in forbidden_keywords:
        if query_upper.startswith(keyword):
            return False
        # Also check for keywords after whitespace/semicolon
        if f' {keyword} ' in query_upper or f';{keyword}' in query_upper:
            return False
    
    # Ensure query starts with SELECT (after cleaning)
    if not query_upper.startswith('SELECT'):
        return False
    
    # Check for SQL injection patterns
    dangerous_patterns = [
        '--', '/*', '*/', 'UNION', 'EXEC', 'EXECUTE', 
        'SP_', 'XP_', 'OPENROWSET', 'OPENDATASOURCE'
    ]
    
    for pattern in dangerous_patterns:
        if pattern in query_upper:
            return False
    
    return True

def cleanSQLQuery(query):
    """Clean and validate SQL query with additional security checks"""
    # Remove any markdown formatting
    query = re.sub(r'```sql\s*', '', query)
    query = re.sub(r'```\s*', '', query)
    query = re.sub(r'^\s*SQL:\s*', '', query, flags=re.IGNORECASE)
    
    # Remove extra whitespace and newlines
    query = ' '.join(query.split())
    
    # Remove trailing explanatory text after semicolon
    if ';' in query:
        query = query.split(';')[0] + ';'
    
    # Ensure query ends with semicolon if it doesn't have one
    if not query.strip().endswith(';'):
        query = query.strip() + ';'
    
    # Final security check
    if not is_safe_query(query):
        return "FORBIDDEN_OPERATION: Only SELECT queries are allowed."
    
    return query

def runQuery(query):
    """Execute SQL query with enhanced security checks"""
    if not hasattr(st.session_state, 'db') or st.session_state.db is None:
        return "Please connect to database first"
    
    # Check if operation is forbidden
    if query.startswith("FORBIDDEN_OPERATION"):
        return "❌ Operation not allowed. Only data retrieval (SELECT) queries are permitted for security reasons."
    
    try:
        # Clean and validate the query
        clean_query = cleanSQLQuery(query)
        
        # Double-check security
        if not is_safe_query(clean_query):
            return "❌ Security check failed. Only SELECT queries are allowed."
        
        logger.info(f"Executing safe query: {clean_query}")
        result = st.session_state.db.run(clean_query)
        return result
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return f"Query execution failed: {str(e)}"

def getResponseForQueryResult(question, query, result):
    """Generate natural language response from query results"""
    template = """You are a helpful database assistant. Provide a clear, natural language response based on the query results.

DATABASE SCHEMA:
{schema}

USER QUESTION: {question}
SQL QUERY EXECUTED: {query}
QUERY RESULT: {result}
Result : {result}

INSTRUCTIONS:
1. Give a clear, conversational response
2. Include specific numbers/data from results
3. If result is empty, say "No records found"
4. If there's an error, explain it simply
5. Be concise but informative
6. Don't show the SQL query:
7. Don't repeat the SQL query in your response

RESPONSE FORMAT EXAMPLES:
- For counts: "There are 25 customer in the database."
- For lists: "I found 3 products: Product A, Product B, and Product C."
- For empty results: "No customer were found matching your criteria."
- For errors: "I couldn't execute the query due to an error with the column name."

    question: how many users above are from india we have in database
    SQL query: SELECT COUNT(*) FROM customer WHERE country=india;
    Result : [(4,)]
    Response: There are 4 amazing users in the database.

Your response:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": question,
            "schema": getDatabaseSchema(),
            "query": query,
            "query_result": str(result),
            "result": str(result)
        })
        
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return f"I encountered an error while processing the results: {str(e)}"

st.title("CA ML Chat with Database")

# Initialize session state
if "chat" not in st.session_state:
    st.session_state.chat = []

if "db_connected" not in st.session_state:
    st.session_state.db_connected = False

# Display CUDA status
cuda_available, cuda_message = check_cuda_status()
# if cuda_available:
#     st.success(f" GPU Status: {cuda_message}")
# else:
#     st.warning(f" GPU Status: {cuda_message} - Running on CPU")

# Main chat interface
question = st.chat_input('Ask a question ...')

if question:
    if not st.session_state.db_connected:
        st.error(' Please connect to the database first using the sidebar.')
    else:
        # Add user message to chat
        st.session_state.chat.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Generating response..."):
            # Generate SQL query
            query = getQueryFromLLM(question)
            
        with st.spinner("Generating response..."):
            # Execute query
            result = runQuery(query)
            
        with st.spinner("Generating response..."):
            # Generate natural language response
            response = getResponseForQueryResult(question, query, result)
            
        # Add assistant response to chat
        # st.session_state.chat.append({
        #     "role": "assistant",
        #     "content": response,
        #     "query": query,
        #     "result": str(result)
        # })
        st.session_state.chat.append({
            "role": "assistant",
            "content": response,
            "query": query,
            "query_result": str(result),
            "result": str(result)
        })

# Display chat history
for i, chat in enumerate(st.session_state.chat):
    with st.chat_message(chat['role']):
        # Display the assistant response (now including the result)
        st.markdown(chat['content'])

        # Ensure result also appears
        # if chat['role'] == 'assistant' and 'result' in chat:
        #     if chat['result']:
        #         st.markdown(f"**Result:** {chat['result']}")
            
# for i, chat in enumerate(st.session_state.chat):
#     with st.chat_message(chat['role']):
#         st.markdown(chat['content'])
        
#         #Show SQL query and result for assistant messages
        # if chat['role'] == 'assistant' and 'query' in chat:
        #     # with st.expander("View SQL Query & Results"):
        #         st.code(chat['query'], language='sql')
        #         st.text("Result:")
        #         st.markdown(chat['result'])
            


# Sidebar for database connection
with st.sidebar:
    st.title(' Database Connection')
    
    # Connection status
    if st.session_state.db_connected:
        st.success(" Database Connected")
    else:
        st.warning(" Not Connected")
    
    # Ensure session state variable exists
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False

# Try to connect on startup
if not st.session_state.db_connected:
    with st.spinner("Connecting to database..."):
        success, message = connectDatabase("root", "3306", "localhost", "Nampraki123*", "rag_test")

    if success:
        st.session_state.db_connected = True
        st.success(message)
        st.rerun()  # Use experimental_rerun() for stability
    else:
        st.session_state.db_connected = False
        st.error(message)
