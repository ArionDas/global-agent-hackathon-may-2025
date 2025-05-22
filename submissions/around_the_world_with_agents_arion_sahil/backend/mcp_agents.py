import os
import asyncio
import sys
from textwrap import dedent
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools, MultiMCPTools
from agno.utils.pprint import apprint_run_response

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def extract_text_from_response(response_stream):
    """
    Extract text content from the response stream object
    """
    try:
        if hasattr(response_stream, 'content'):
            return response_stream.content
        elif hasattr(response_stream, 'messages') and response_stream.messages:
            # Get the last message content
            last_message = response_stream.messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                return last_message['content']
        elif hasattr(response_stream, 'text'):
            return response_stream.text
        elif isinstance(response_stream, str):
            return response_stream
        else:
            # Try to convert to string as fallback
            return str(response_stream)
    except Exception as e:
        print(f"Error extracting text from response: {e}")
        return None

## Transport Agent
async def transport_mcp_agent(message: str, people: int = 1):
    response_text = None
    try:
        async with MCPTools("npx -y @modelcontextprotocol/server-google-maps") as mcp_tools:
            
            print("MCPTools initializing for Transport Agent...")
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
                instructions=dedent(f"""\
                    You are a travel agent. Your task is to find the best transport options for {people} number of people.\
                    You will find the time and cost of traveling from one location to another - for each of the following modes:\
                    - Car\
                    - Train\
                    - Flight\
                    You will use the Google Maps API to find the best transport options.\
                    You will return the response in JSON format.\
                    You will not return any unnecessary text.
                    """
                ),
                tools=[mcp_tools],
                markdown=True,
            )
            print("Transport Agent initialized!")
            
            try:
                # Get response without streaming for easier text extraction
                response_stream = await agent.arun(message, stream=False)
                
                # Print the response for debugging
                await apprint_run_response(response_stream, markdown=True)
                
                # Extract the actual text content
                response_text = extract_text_from_response(response_stream)
                
            finally:
                # Attempt to clean up agent if it has a proper close method
                if hasattr(agent, 'close') and callable(agent.close):
                    try:
                        if asyncio.iscoroutinefunction(agent.close):
                            await agent.close()
                        else:
                            agent.close()
                        print("Agent closed")
                    except Exception as e:
                        print(f"Error closing agent: {e}")
        
        print("MCPTools context exited.")

    except Exception as e:
        print(f"Error occurred in transport_agent: {e}")
    
    return response_text


## Hotel Booking Agent
async def hotel_booking_mcp_agent(message: str, place: str, people: int = 1):
    response_text = None
    try:
        async with MultiMCPTools(
            [
                "npx -y @openbnb/mcp-server-airbnb --ignore-robotics-txt",
                "npx -y @modelcontextprotocol/server-google-maps"
            ]
        ) as mcptools:
            
            print("MCPTools initializing for Hotel Booking Agent...")
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
                instructions=dedent(f"""\
                    You are a hotel booking assistant. Your task is to find the best hotel options for the user at {place} for {people} number of people.\
                    You will find the cheapest offers which meets the requirements of the user:\
                    You will use the Google Maps API and Airbnb API to find the best shelter options.\
                    You will return the response in text format.\
                    You will not return any unnecessary text.
                    """
                ),
                tools=[mcptools],
                markdown=True,
            )
            print("Hotel Booking Agent initialized!")
            
            try:
                response_stream = await agent.arun(message, stream=False)
                await apprint_run_response(response_stream, markdown=True)
                
                # Extract the actual text content
                response_text = extract_text_from_response(response_stream)
                
            finally:
                if hasattr(agent, 'close') and callable(agent.close):
                    try:
                        if asyncio.iscoroutinefunction(agent.close):
                            await agent.close()
                        else:
                            agent.close()
                        print("Agent closed")
                    except Exception as e:
                        print(f"Error closing agent: {e}")
                    
        print("MCPTools context exited.")
    
    except Exception as e:
        print(f"Error occurred in hotel_booking_agent: {e}")
        
    return response_text


## Sightseeing Agent
async def sightseeing_mcp_agent(message: str, place: str, people: int = 1):
    response_text = None
    try:
        async with MCPTools("npx -y @modelcontextprotocol/server-google-maps") as mcptools:
            
            print("MCPTools initializing for Sightseeing Agent...")
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
                instructions=dedent(f"""\
                    You are a local tour guide. Your task is to find the best sightseeing locations in and around {place}.\
                    There are {people} number of people. Don't come up with places too far.\
                    You will use the Google Maps API to find the best sightseeing options.\
                    You will return the response in text format.\
                    You will not return any unnecessary text.
                    """
                ),
                tools=[mcptools],
                markdown=True,
            )
            print("Sightseeing Agent initialized!")
            
            try:
                response_stream = await agent.arun(message, stream=False)
                await apprint_run_response(response_stream, markdown=True)
                
                # Extract the actual text content
                response_text = extract_text_from_response(response_stream)
                
            finally:
                if hasattr(agent, 'close') and callable(agent.close):
                    try:
                        if asyncio.iscoroutinefunction(agent.close):
                            await agent.close()
                        else:
                            agent.close()
                        print("Agent closed")
                    except Exception as e:
                        print(f"Error closing agent: {e}")
                    
        print("MCPTools context exited.")
    
    except Exception as e:
        print(f"Error occurred in sightseeing_agent: {e}")
        
    return response_text


## Location Agent
async def location_mcp_agent(message: str, place: str, days_left: int):
    response_text = None
    try:
        async with MCPTools("npx -y @modelcontextprotocol/server-google-maps") as mcptools:
            
            print("MCPTools initializing for Location Agent...")
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
                instructions=dedent(f"""\
                    You are a travel agent. The tourists have {days_left} days left.\
                    You will find the next most popular location for the user to go to from {place}.\
                    You will use the Google Maps API to find the best location.\
                    You will return the response in text format.\
                    You will not return any unnecessary text.
                    """
                ),
                tools=[mcptools],
                markdown=True,
            )
            print("Location Agent initialized!")
            
            try:
                response_stream = await agent.arun(message, stream=False)
                await apprint_run_response(response_stream, markdown=True)
                
                # Extract the actual text content
                response_text = extract_text_from_response(response_stream)
                
            finally:
                if hasattr(agent, 'close') and callable(agent.close):
                    try:
                        if asyncio.iscoroutinefunction(agent.close):
                            await agent.close()
                        else:
                            agent.close()
                        print("Agent closed")
                    except Exception as e:
                        print(f"Error closing agent: {e}")
                    
        print("MCPTools context exited.")
    
    except Exception as e:
        print(f"Error occurred in location_agent: {e}")
        
    return response_text


# Test function to demonstrate usage
async def test_agents():
    """
    Test function to show how all agents work and return text
    """
    print("=== Testing Transport Agent ===")
    transport_result = await transport_mcp_agent(
        "Find me the price of traveling from Kolkata to Sikkim using car, train, flight. Please return in json format.",
        people=2
    )
    print(f"Transport Result Type: {type(transport_result)}")
    print(f"Transport Result: {transport_result}")
    
    print("\n=== Testing Hotel Booking Agent ===")
    hotel_result = await hotel_booking_mcp_agent(
        "Find me affordable hotels in Sikkim for 2 people for 3 nights",
        place="Sikkim",
        people=2
    )
    print(f"Hotel Result Type: {type(hotel_result)}")
    print(f"Hotel Result: {hotel_result}")
    
    print("\n=== Testing Sightseeing Agent ===")
    sightseeing_result = await sightseeing_mcp_agent(
        "What are the top 5 sightseeing places in Sikkim?",
        place="Sikkim",
        people=2
    )
    print(f"Sightseeing Result Type: {type(sightseeing_result)}")
    print(f"Sightseeing Result: {sightseeing_result}")
    
    print("\n=== Testing Location Agent ===")
    location_result = await location_mcp_agent(
        "What's the next best place to visit from Sikkim for a 7-day trip?",
        place="Sikkim",
        days_left=4
    )
    print(f"Location Result Type: {type(location_result)}")
    print(f"Location Result: {location_result}")
            

if __name__ == "__main__":
    print("Starting main execution...")
    
    try:
        # Test single agent
        message = "Find me the price of traveling from Kolkata to Sikkim using car, train, flight. Please return in json format, no unnecessary text to be returned."
        final_response = asyncio.run(transport_mcp_agent(message, people=2))
        
        if final_response:
            print(f"Transport agent response type: {type(final_response)}")
            print(f"Transport agent response: {final_response}")
        else:
            print("No response received from transport_agent or an error occurred.")
            
        # Uncomment to test all agents:
        # print("\n" + "="*50)
        # print("Testing all agents...")
        # asyncio.run(test_agents())
    
    except Exception as e:
        print(f"Error occurred in main execution block: {e}")
    finally:
        print("Main execution finished.")