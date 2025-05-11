import os
import asyncio
from dotenv import load_dotenv
from agents_arion import team_leader, transport_agent
from agents_sahil import transport_agent, location_agent, sightseeing_agent, hotel_booking_agent

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        
        transport = transport_agent.run(
            message = f"Find me the price of traveling from {start} to {end} using car, train, flight. Please return in json format, no unnecessary text to be returned."
        )
        print(transport)
        
        ## Step 2: Hotel Booking Agent :: at destination
        hotel = ""
        
        hotel = hotel_booking_agent.run(
            message = f"Show me the 2 cheapest hotels for the location {end}."
        )
        print(hotel)
        
        ## Step 3: Sightseeing Agent :: 
        # give user option to choose number of days (maybe)
        
        sightseeing = ""
        
        sightseeing = sightseeing_agent.run(
            message = f"Find me the 4 best sightseeing options for the location {end}."
        )
        print(sightseeing)
        
        ## Step 4: Update start and end & append everything to complete LLM prompt.
        start = end
        end = location_agent.run(
            message = f"Find me the next best tourist destination closest from {start}."
        )
        
        total_prompt += f"Day: {day}\nTransport: {transport}\nHotel: {hotel}\nSightseeing: {sightseeing}\n"
        
        total_days -= 1
        print(f"Total days left: {total_days}")
        
    return total_prompt
        

if __name__ == "__main__":
    start_location = "New York"
    tourist_destination = "Los Angeles"
    end_location = "Chicago"
    budget = 1000
    total_days = 5
    number_of_people = 2
    
    prompt = multi_agent_collaboration(start_location, tourist_destination, end_location, budget, total_days, number_of_people)
        
    print(prompt)

        