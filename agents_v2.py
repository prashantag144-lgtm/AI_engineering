# STEP 1 : IMPORTING ALL THE LIBRARIES AND FRAMEWORK

from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,ToolMessage
from tavily import TavilyClient
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.middleware import wrap_tool_call
from langchain.agents import create_agent

# STEP 2: CREATION OF TOOLS

# Weather tool

@tool
def get_weather(city:str)->str:
    """Get the current weather for the given city"""
    API_KEY=os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response=requests.get(url)
    data=response.json()
    """ pasrer=StrOutputParser()
    print("DEBUG:",data) """

    if str(data.get("cod"))!="200":
        return f"Error : {data.get('message','Could not fetch weather')}"

    temp=data["main"]["temp"]
    desc=data["weather"][0]["description"]

    return f"weather in {city}: {desc}, {temp} degreee celsius" 


# Tavily new tool

tavily_client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(news:str)->str:
    """Get the current news of the given query"""
    city="bengaluru"
    query=f"latest news in {city}"
    response=tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=3
    )

    result=response.get("results",[])
    if not result:
            return f"No recent news found for {city}"
            
    news_list=[]
    for r in result:
            url=r.get("url","")
            snippet=r.get("content","")
    
            news_list.append(
                f" {url} {snippet[:100]}"
            )
    
    return f"latest news in {city}:" + "".join(news_list)

    
llm=ChatGroq(model="openai/gpt-oss-120b", api_key="")


@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  

agent = create_agent(
    llm,
    tools = [get_weather,get_news],
    system_prompt= "you are a helpful city assistant.",
    middleware= [human_approval]
)

print("City Agent | type exit to quit")
history=[]
while True:
    user_input = input("You : ")
    history.append(user_input)
    if user_input.lower() == "exit":
        break 
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    print("bot : ", result['messages'][-1].content )
    history.append(result['messages'][-1].content)
        
