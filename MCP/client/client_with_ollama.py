import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
#Using Ollama
model = ChatOllama(model="llama3.2")

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["../server/string_tools_server.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            # Try out the tools via natural language
            msg1 = {"messages": "can you invert the string 'Harry'"}
            msg2 = {"messages": "How many words are in the sentence 'kavreli samdhi k chha khabar, thik chha ni?'?"}
            res1 = await agent.ainvoke(msg1)
            # print("Reversed string result:", res1)
            for m in res1['messages']:
                m.pretty_print()
            res2 = await agent.ainvoke(msg2)
            # print("Word count result:", res2)
            for m in res2['messages']:
                m.pretty_print()
if __name__ == "__main__":
    asyncio.run(main())