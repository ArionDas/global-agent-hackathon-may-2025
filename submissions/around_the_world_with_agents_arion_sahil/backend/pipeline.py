import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agents_arion import team_leader, transport_agent
from agents_sahil import transport_agent, location_agent, sightseeing_agent, hotel_booking_agent
from mcp_agents import transport_mcp_agent, hotel_booking_mcp_agent, sightseeing_mcp_agent, location_mcp_agent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def multi_agent_collaboration(start_location: str, tourist_destination: str, end_location: str, budget: float, total_days: int, number_of_people: int):
    
    """
    This function coordinates the multi-agent collaboration for a travel itinerary.
    It hard codes the flow between individual agents and their respective tasks.
    We didn't want some pre-defined "team agent" to have the control, instead we wanted to have a more flexible approach.
    """
    
    assert(total_days > 0), "Total days must be greater than 0"
    
    assert(budget > 0), "Budget must be greater than 0"
    
    assert(number_of_people > 0), "Number of people must be greater than 0"
    
    assert(start_location != tourist_destination), "Start location and tourist destination must be different"
    
    start = start_location
    end = tourist_destination
    
    total_prompt = ""
    day = 0
    
    while(total_days > 0):
        
        day += 1
        
        ## Step 1: Transport Agent :: start -> destination
        transport = ""
        
        transport = asyncio.run(transport_mcp_agent(
            message = f"Find me the price of traveling from {start} to {end} using car, train, flight. Please return in json format, no unnecessary text to be returned.",
            people = number_of_people,
        ))
        print("Transport : ", transport)
        
        
        ## Step 2: Sightseeing Agent :: 
        # give user option to choose number of days (maybe)
        
        sightseeing = ""
        
        sightseeing = asyncio.run(sightseeing_mcp_agent(
            message = f"Find me the 4 best sightseeing options for the location {end}. Be very specific and give me the best options.",
            place = end,
            people = number_of_people,
        ))
        print("Sightseeing: ", sightseeing)
        
        
        ## Step 3: Update start and end locations.
        start = end
        end = asyncio.run(location_mcp_agent(
            message = f"Find me the nearest most popular tourist destination closest from {start} where tourists can spend the night.",
            place = start,
            days_left = total_days - 1,
        ))
        print("Next destination: ", end)
        
        
        ## Step 4: Hotel Booking Agent ::
        hotel = ""
        hotel = asyncio.run(hotel_booking_mcp_agent(
            message = f"Find me the best hotel in {end} for {number_of_people} people. Be very specific and give me the best options.",
            place = end,
            people = number_of_people,
        ))
        print("Hotel: ", hotel)
        
        
        ## Step 5: Append eveyrthing to the final prompt.
        total_prompt += f"Day: {day}\nTransport: {transport}\nHotel: {hotel}\nSightseeing: {sightseeing}\n"
        
        total_days -= 1
        print(f"Total days left: {total_days}")
        
    return total_prompt
        

if __name__ == "__main__":
    
    st.set_page_config(layout='wide')
    st.title("Around the World with Agents")
    st.write("This is a collaborative multi-agent application for travel itinerary planning.")
    
    with st.sidebar:
        st.header("API Keys:")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        groq_api_key = st.text_input("Groq API Key", type="password")
        google_maps_api_key = st.text_input("Google Maps API Key", type="password")
        if not openai_api_key or not groq_api_key or not google_maps_api_key:
            st.warning("Please enter all API keys to proceed.")
            
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["GOOGLE_MAPS_API_KEY"] = google_maps_api_key
        
        st.header("Travel Details:")
        start_location = st.text_input("Start Location")
        tourist_destination = st.text_input("Tourist Destination")
        end_location = st.text_input("End Location")
        budget = st.number_input("Budget (in INR)")
        total_days = st.number_input("Total Days")
        number_of_people = st.number_input("Number of People")
        if not start_location or not tourist_destination or not end_location or budget <= 0 or total_days <= 0 or number_of_people <= 0:
            st.warning("Please fill all travel details to proceed.")
    
    if st.button("Generate Itinerary"):
        prompt = multi_agent_collaboration(start_location, tourist_destination, end_location, budget, total_days, number_of_people)
    
        llm_agent = Agent(model = Groq("llama-3.3-70b-versatile"), markdown = True)
        
        itinerary = llm_agent.print_response(f"""Summarize the trip for the tourists. They are a group of {number_of_people} people traveling from {start_location} to {tourist_destination} and back to {end_location}.
                                The trip is for {total_days} days. The budget is {budget}.
                                
                                You have to provide a detailed itinerary for the trip.
                                The itinerary should include the following:
                                    - Day-wise breakdown of the trip
                                    - Transport options with prices
                                    - Hotel options with prices
                                    - Sightseeing options with prices
                                    - Any other relevant information that you think is important for the trip.
                                
                                Here is the trip plan:
                                {prompt}.
                                
                                If you feel anything amiss, please feel free to add your own knowledge.""")
        
        st.subheader("Generated Itinerary:")
        st.markdown(itinerary, unsafe_allow_html=True)

        