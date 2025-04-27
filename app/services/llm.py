import os
import json
from typing import Dict, List, Any, Optional
import logging
import httpx
from openai import OpenAI
from app.core.config import settings
import re

async def generate_tasks(
    environment_data: Dict[str, Any], 
    scenario_data: Dict[str, Any],
    num_tasks: int = 5,
    task_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate logistics tasks using a large language model.
    
    Args:
        environment_data: Data about the warehouse environment
        scenario_data: Data about the scenario
        num_tasks: Number of tasks to generate
        task_type: Optional type of tasks to generate (pickup, delivery, etc.)
    
    Returns:
        List of generated tasks
    """
    try:
        # Build the prompt
        prompt = build_task_generation_prompt(
            environment_data, 
            scenario_data, 
            num_tasks, 
            task_type
        )
        
        # Call LLM API
        response = await call_llm_api(prompt)
        
        # Parse and validate tasks
        tasks = parse_and_validate_tasks(response, environment_data)
        
        return tasks
    
    except Exception as e:
        logging.error(f"Error generating tasks with LLM: {str(e)}")
        return []


def build_task_generation_prompt(
    environment_data: Dict[str, Any], 
    scenario_data: Dict[str, Any], 
    num_tasks: int,
    task_type: Optional[str]
) -> str:
    """
    Build a prompt for task generation.
    
    Args:
        environment_data: Data about the warehouse environment
        scenario_data: Data about the scenario
        num_tasks: Number of tasks to generate
        task_type: Optional type of tasks to generate
    
    Returns:
        Generated prompt string
    """
    # Get relevant data from environment
    dimensions = environment_data.get("dimensions", {})
    elements = environment_data.get("elements", {})
    
    # Get shelves and dropoff points
    shelves = elements.get("shelves", [])
    dropoffs = elements.get("dropoffs", [])
    robot_stations = elements.get("robot_stations", [])
    robots = elements.get("robots", [])
    pickups = elements.get("pickups", [])
    
    # Format environment data for the prompt
    env_description = (
        f"Warehouse dimensions: {dimensions.get('width')}x{dimensions.get('height')} units\n"
        f"Number of shelves: {len(shelves)}\n"
        f"Number of dropoff points: {len(dropoffs)}\n"
        f"Number of pickups: {len(pickups)}\n"
        f"Number of robots: {len(robots)}\n"
        f"Pickup points with shelf ids: {pickups}\n"
        f"Dropoff points with shelf ids: {dropoffs}\n"
    )
    
    # Format scenario data
    scenario_description = (
        f"Scenario: {scenario_data.get('name')}\n"
        f"Description: {scenario_data.get('description')}\n"
    )
    
    # Add scenario parameters
    # scenario_params = scenario_data.get("parameters", {})
    # scenario_description += "Parameters:\n"
    # for key, value in scenario_params.items():
    #     scenario_description += f"- {key}: {value}\n"
    
    # Build the task type constraint
    task_type_constraint = ""
    # if task_type:
    #     task_type_constraint = f"Generate only {task_type} tasks."
    
    # Full prompt
    prompt = f"""
    You are a warehouse logistics task generator. Generate {num_tasks} realistic logistics tasks for a warehouse robot system.

    WAREHOUSE ENVIRONMENT:
    {env_description}

    SCENARIO:
    {scenario_description}

    For each task, please provide:


    Return the tasks as json in the following format:
    [
        [
            "P1", "D2"
        ],
        [
            "P2", "D3"
        ],
        ...
    ]

    Ensure all points are within the warehouse dimensions and reference valid locations.
    """
    
    return prompt


async def call_llm_api(prompt: str) -> str:
    """
    Call the OpenAI API with a prompt.
    
    Args:
        prompt: The prompt to send to the LLM
    
    Returns:
        LLM response string
    """
    try:
        # Set the API key
        api_key = settings.OPENAI_API_KEY
        client = OpenAI(api_key=api_key)
        # Make the request to OpenAI
        completion = client.chat.completions.create(
            model="gpt-4.1",  # Or use a different model like "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a warehouse logistics task generator. Generate realistic logistics tasks as requested in the format specified."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        # Extract and return the response text
        response_content = completion.choices[0].message.content
        print(f"Response content: {response_content}")
        return response_content
    
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {str(e)}")
        
        # Fallback to placeholder if API fails
        logging.warning("Falling back to placeholder task generation")
        example_tasks = [
            {
                "name": "Pickup Order #12345",
                "description": "Collect package from shelf A3 for customer order",
                "task_type": "pickup_delivery",
                "start_point": [2, 3],
                "end_point": [10, 15],
                "priority": 3,
                "additional_details": {"order_id": "12345", "customer": "John Doe"}
            },
            {
                "name": "Deliver to Packing Station",
                "description": "Deliver collected items to packing station 2",
                "task_type": "pickup_delivery",
                "start_point": [10, 15],
                "end_point": [20, 5],
                "priority": 4,
                "additional_details": {"packing_station_id": "PS2"}
            }
        ]
        
        return json.dumps(example_tasks)


def parse_and_validate_tasks(llm_response: str, environment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse and validate tasks from LLM response.
    
    Args:
        llm_response: Response from the LLM
        environment_data: Environment data for validation
    
    Returns:
        List of validated tasks
    """
    try:
        # Try to extract JSON from the response
        # Sometimes the model might include additional text before or after the JSON
        pickups = environment_data.get("elements", {}).get("pickups", [])
        dropoffs = environment_data.get("elements", {}).get("dropoffs", [])
        pickups_ids = [pickup.get("id") for pickup in pickups]
        dropoffs_ids = [dropoff.get("id") for dropoff in dropoffs]
        
        json_match = re.search(r'\[[\s\S]*\]', llm_response)
        
        if json_match:
            json_str = json_match.group(0)
            tasks = json.loads(json_str)
        else:
            # If we can't find a JSON array in brackets, try parsing the whole response
            tasks = json.loads(llm_response)
        
        # Ensure it's a list
        if not isinstance(tasks, list):
            raise ValueError("LLM response is not a list of tasks")
        
        # Validate each task
        validated_tasks = {"tasks": []}
        for task in tasks:
            # Check required fields
            print(f"task: {task}")
            if len(task) != 2:
                logging.warning(f"Task has invalid format: {task}")
                continue
            
            if task[0] not in pickups_ids or task[1] not in dropoffs_ids:
                logging.warning(f"Task has invalid pickup or dropoff id: {task}")
                continue
            # Add task to validated list
            validated_tasks["tasks"].append(task)
        
        if not validated_tasks:
            logging.error("No valid tasks could be extracted from LLM response")
            return []
        
        return validated_tasks
    
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response as JSON: {e}")
        logging.debug(f"Raw response: {llm_response}")
        return []
    except Exception as e:
        logging.error(f"Error validating tasks: {str(e)}")
        return []