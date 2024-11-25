import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()

# Inicializar agentes
def initialize_agents():
    print("Inicializando agentes...")
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    print("Base prompt cargado.")

    # Agente Python
    python_instructions = """
    You are an agent designed to write and execute Python code to answer questions.
    You have access to a Python REPL, which you can use to execute Python code.
    Your job is to interpret questions, generate the appropriate Python code, and execute it to get an answer.
    If you cannot execute the code or it fails, debug and retry.
    """
    python_prompt = base_prompt.partial(instructions=python_instructions)
    python_tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=python_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=python_tools,
    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=python_tools, verbose=True)
    print("Agente Python inicializado.")  # Debug statement

    # Agente CSV
    csv_files = [
        "rankings_history.csv",
        "fighter_stats.csv",
        "UFC_Fights.csv",
        "ufc_ppv_buys.csv",
    ]
    csv_agents = {
        file: create_csv_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-4"),
            path=file,
            verbose=True,
            allow_dangerous_code=True,
        )
        for file in csv_files
    }
    print("Agente CSV inicializado.")

    # Tools
    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor.invoke,
            description="""Useful when you need to transform natural language to Python 
            and execute the Python code, returning the results of the code execution.
            This tool expects natural language input, NOT raw Python code."""
        ),
    ]
    tools += [
        Tool(
            name=f"CSV Agent ({file})",
            func=csv_agents[file].invoke,
            description=f"Useful for questions about the content of {file}.",
        )
        for file in csv_files
    ]

    grand_prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=grand_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(
        agent=grand_agent,
        tools=tools,
        verbose=True,
    )

    print("Todos los agentes inicializados y listos.")  # Debug statement
    return grand_agent_executor

def main():
    st.title("Agente Inteligente con Python y CSV")

    # Inicializar el agente
    st.write("Cargando agentes...")
    grand_agent_executor = initialize_agents()
    st.success("Agentes listos para usar.")

    # Menú para Python Agent
    st.subheader("Selecciona una tarea para el Python Agent")
    python_tasks = st.multiselect(
        "Elige las solicitudes:",
        options=[
            "Imprime hola mundo",
            "1+1",
            "Dame el codigo para crear un diccionario vacio en python",
        ],
    )
    if st.button("Ejecutar tareas seleccionadas"):
        for task in python_tasks:
            st.write(f"Enviando al agente: {task}")
            try:
                result = grand_agent_executor.invoke({"input": task})
                st.write(f"Resultado para '{task}': {result}")
            except Exception as e:
                st.error(f"Error al procesar '{task}': {str(e)}")

    # Campo de texto para preguntas
    st.subheader("Pregunta sobre los CSV o solicita generación de un programa")
    user_query = st.text_input("Escribe tu pregunta aquí:")
    if st.button("Procesar pregunta"):
        if user_query.strip():
            st.write(f"Enviando al agente: {user_query}")
            try:
                result = grand_agent_executor.invoke({"input": user_query})
                st.write("Respuesta del agente:", result)
            except Exception as e:
                st.error(f"Error al procesar la pregunta: {str(e)}")
        else:
            st.warning("Por favor, escribe una pregunta antes de presionar el botón.")

if __name__ == "__main__":
    main()
