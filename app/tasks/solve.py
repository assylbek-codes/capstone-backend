import importlib
import logging
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.db.base import SessionLocal
from app.models.solve import Solve, SolveStatus
from app.worker import celery_app

import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import random
import numpy as np

# Set up more detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("robot_scheduler")

def build_undirected_graph_from_json(env_json):
    G = nx.Graph()
    added = set()
    for node in env_json['nodes']:
        G.add_node(node['id'])
    for edge in env_json['edges']:
        # Only add edge if not already added in either direction
        key = tuple(sorted([edge['from'], edge['to']]))
        if key not in added:
            G.add_edge(edge['from'], edge['to'], weight=edge['weight'])
            added.add(key)
    return G

@celery_app.task(bind=True, name='app.tasks.solve.execute_solve')
def execute_solve(self, solve_id: int) -> Dict[str, Any]:
    """
    Task to execute a solve operation.
    
    Args:
        solve_id: ID of the solve record in the database
    
    Returns:
        Dictionary with results of the solve
    """
    print(f"STARTING EXECUTE_SOLVE TASK for solve_id={solve_id}")
    logging.info(f"Starting solve execution for solve_id={solve_id}")
    
    # Create DB session
    db = SessionLocal()
    try:
        # Get the solve record
        solve = db.query(Solve).filter(Solve.id == solve_id).first()
        if not solve:
            logging.error(f"Solve with ID {solve_id} not found")
            return {"status": "error", "message": f"Solve with ID {solve_id} not found"}
        
        # Update status to running
        solve.status = SolveStatus.RUNNING
        db.commit()
        
        # Get all needed data for the solve
        environment = solve.environment
        task = solve.task
        algorithm = solve.algorithm
        
        # Log the parameters
        logger.info(f"Solve parameters: {solve.parameters}")
        battery_capacity = solve.parameters.get('battery_capacity', float('inf')) if solve.parameters else float('inf')
        max_distance = solve.parameters.get('max_distance', float('inf')) if solve.parameters else float('inf')
        recharge_time = solve.parameters.get('recharge_time', 0) if solve.parameters else 0
        logger.info(f"Battery parameters: capacity={battery_capacity}, max_distance={max_distance}, recharge_time={recharge_time}")
        
        try:
            # Import the algorithm module and function
            module = importlib.import_module(algorithm.module_path)
            algo_function = getattr(module, algorithm.function_name)
            robots = environment.elements.get("robots")
            robots_ids = [r.get("id") for r in robots]
            nx_graph = build_undirected_graph_from_json(environment.graph)
            
            logger.info(f"Starting algorithm: {algorithm.function_name}")
            logger.info(f"Task count: {len(task.details.get('tasks', []))}")
            logger.info(f"Robot count: {len(robots_ids)}")
            
            # Execute the algorithm with parameters
            result = algo_function(
                tasks=task.details.get("tasks"),
                robots=robots_ids,
                graph=nx_graph,
                env_json=environment.graph,
                params=solve.parameters  # Pass parameters from frontend
            )
            
            # Log some statistics from the result
            total_tasks = sum(len([t for t in tasks if isinstance(t.get('task'), tuple)]) 
                             for robot, tasks in result.items())
            total_recharges = sum(len([t for t in tasks if t.get('task') == 'recharge' or t.get('task') == 'recharge_in_place']) 
                                for robot, tasks in result.items())
            
            logger.info(f"Algorithm completed: {total_tasks} tasks assigned, {total_recharges} recharges scheduled")
            
            # Update the solve record with results
            solve.results = result
            solve.status = SolveStatus.COMPLETED
            db.commit()
            
            return {
                "status": "success",
                "solve_id": solve_id,
                "result": result
            }
            
        except Exception as e:
            logging.error(f"Error executing algorithm: {str(e)}")
            solve.status = SolveStatus.FAILED
            db.commit()
            return {
                "status": "error",
                "message": f"Algorithm execution failed: {str(e)}",
                "solve_id": solve_id
            }
    
    except Exception as e:
        logging.error(f"Error in solve execution: {str(e)}")
        return {"status": "error", "message": str(e)}
    
    finally:
        db.close() 


def compute_cost(graph, start, pickup, dropoff, params=None):
    """Compute cost = distance(start→pickup) + distance(pickup→dropoff) factoring in robot speed."""
    cost1 = nx.shortest_path_length(graph, start, pickup, weight='weight')
    cost2 = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
    
    # Apply robot speed if available
    if params and 'robot_speed' in params:
        robot_speed = params.get('robot_speed', 1.0)
        if robot_speed > 0:
            # Convert distance to time cost based on speed
            return (cost1 + cost2) / robot_speed
    
    return cost1 + cost2

def get_robot_station(robot_id):
    """Extract the charging station ID from a robot ID (e.g., R1_1 -> R1)."""
    if '_' in robot_id:
        return robot_id.split('_')[0]
    return robot_id  # If no underscore, use the ID as is

def ensure_station_exists(graph, station):
    """Ensure charging station exists in the graph, add it if not."""
    if station not in graph.nodes:
        # Add the station with connections to all nodes starting with 'N'
        graph.add_node(station)
        # Connect to nearby nodes to make it reachable
        for node in graph.nodes:
            if node.startswith('N'):
                graph.add_edge(station, node, weight=1.0)
    return station

def calculate_battery_remaining(battery_capacity, max_distance, distance_traveled):
    """Calculate battery remaining as a percentage based on distance traveled relative to max distance."""
    logger.debug(f"Calculating battery: capacity={battery_capacity}, max_distance={max_distance}, distance_traveled={distance_traveled:.2f}")
    
    if max_distance == float('inf') or battery_capacity == float('inf'):
        return battery_capacity  # If no limit, always return full capacity
    
    # Calculate battery percentage (linear relationship between distance and battery)
    # Example: if max_distance is 10000 and distance_traveled is 1000, battery is reduced by 10%
    battery_percentage = 1.0 - (distance_traveled / max_distance)
    battery_percentage = max(0.0, min(1.0, battery_percentage))  # Clamp between 0 and 1
    
    # Convert percentage to actual battery level
    battery_remaining = battery_percentage * battery_capacity
    logger.debug(f"Battery result: {battery_remaining:.2f}% ({battery_percentage:.2%})")
    
    return battery_remaining

def greedy_assignment(tasks, robots, graph, env_json, params=None):
    """1. Greedy Distance Minimization with robot parameters."""
    assignment = {r: [] for r in robots}
    robots_positions = {r: r for r in robots}
    
    # Get robot parameters
    battery_capacity = params.get('battery_capacity', float('inf')) if params else float('inf')
    max_distance = params.get('max_distance', float('inf')) if params else float('inf')
    recharge_time = params.get('recharge_time', 0) if params else 0
    
    # Track battery and distance for each robot
    robot_battery = {r: battery_capacity for r in robots}
    robot_distance = {r: 0 for r in robots}
    # Track time for each robot (for scheduling with recharge time)
    robot_time = {r: 0 for r in robots}
    
    for pickup, dropoff in tasks:
        # Find robot with minimum cost, accounting for battery and max distance
        valid_robots = [r for r in robots if robot_battery[r] > 0 and robot_distance[r] < max_distance]
        if not valid_robots:
            # Recharge robots at their stations
            for r in robots:
                if robot_battery[r] <= 0:
                    # Get the robot's charging station
                    station = get_robot_station(r)
                    
                    # Path to charging station
                    path_to_station = nx.shortest_path(graph, robots_positions[r], station, weight='weight')
                    station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                    
                    # Add recharging path to assignment
                    assignment[r].append({
                        'task': 'recharge',
                        'path': path_to_station,
                        'battery_remaining': battery_capacity,
                        'total_distance': robot_distance[r] + station_cost,
                        'recharge_time': recharge_time
                    })
                    
                    # Update position, battery, distance and time
                    robots_positions[r] = station
                    robot_battery[r] = battery_capacity
                    robot_distance[r] = 0  # Reset distance traveled counter for battery calc
                    # Add recharge time to robot's schedule
                    robot_time[r] += station_cost + recharge_time
            
            # Recalculate valid robots
            valid_robots = [r for r in robots if robot_battery[r] > 0 and robot_distance[r] < max_distance]
            if not valid_robots:
                valid_robots = robots  # Use all robots if still no valid ones
        
        # Now factor in time since recharging to get the "true" cost
        best_robot = min(valid_robots, key=lambda r: 
                          compute_cost(graph, robots_positions[r], pickup, dropoff, params) + robot_time[r])
        
        # Check if robot needs to recharge before task
        if battery_capacity != float('inf'):
            path_to_pickup = nx.shortest_path(graph, robots_positions[best_robot], pickup, weight='weight')
            path_to_dropoff = nx.shortest_path(graph, pickup, dropoff, weight='weight')
            full_path = path_to_pickup + path_to_dropoff[1:]
            path_cost = nx.path_weight(graph, full_path, weight='weight')
            
            # Calculate battery needed for this path
            battery_needed = path_cost / max_distance * battery_capacity
            
            if robot_battery[best_robot] < battery_needed:
                # Get the robot's charging station
                station = get_robot_station(best_robot)
                
                # Path to charging station
                path_to_station = nx.shortest_path(graph, robots_positions[best_robot], station, weight='weight')
                station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                
                # Add recharging path to assignment
                assignment[best_robot].append({
                    'task': 'recharge',
                    'path': path_to_station,
                    'battery_remaining': battery_capacity,
                    'total_distance': robot_distance[best_robot] + station_cost,
                    'recharge_time': recharge_time
                })
                
                # Update position and battery
                robots_positions[best_robot] = station
                robot_battery[best_robot] = battery_capacity
                robot_distance[best_robot] = 0  # Reset distance traveled counter for battery calc
                # Add recharge time to robot's schedule
                robot_time[best_robot] += station_cost + recharge_time
                
                # Recalculate paths from charging station
                path_to_pickup = nx.shortest_path(graph, robots_positions[best_robot], pickup, weight='weight')
                path_to_dropoff = nx.shortest_path(graph, pickup, dropoff, weight='weight')
                full_path = path_to_pickup + path_to_dropoff[1:]
                path_cost = nx.path_weight(graph, full_path, weight='weight')
        else:
            # Calculate path directly if no battery concerns
            path_to_pickup = nx.shortest_path(graph, robots_positions[best_robot], pickup, weight='weight')
            path_to_dropoff = nx.shortest_path(graph, pickup, dropoff, weight='weight')
            full_path = path_to_pickup + path_to_dropoff[1:]
            path_cost = nx.path_weight(graph, full_path, weight='weight')
        
        # Update robot distance
        robot_distance[best_robot] += path_cost
        
        # Use the new battery calculation function
        robot_battery[best_robot] = calculate_battery_remaining(
            battery_capacity, 
            max_distance, 
            robot_distance[best_robot]
        )
        
        # Update robot time (for scheduling)
        robot_time[best_robot] += path_cost
        
        assignment[best_robot].append({
            'task': (pickup, dropoff), 
            'path': full_path,
            'battery_remaining': robot_battery[best_robot],
            'total_distance': robot_distance[best_robot],
            'completion_time': robot_time[best_robot]
        })
        
        # Update robot's new position for next iteration
        robots_positions[best_robot] = dropoff
    
    return assignment

def hungarian_assignment(tasks, robots, graph, env_json, params=None):
    """
    Assign all tasks among robots using the Hungarian Algorithm in batches.
    Each robot gets one task per batch, and their location is updated after each batch.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import copy

    logger.info(f"Starting hungarian_assignment with {len(tasks)} tasks and {len(robots)} robots")
    logger.info(f"Parameters: battery_capacity={params.get('battery_capacity', 'inf')}, max_distance={params.get('max_distance', 'inf')}, recharge_time={params.get('recharge_time', 0)}")

    robot_locations = {r: r for r in robots}
    assignments = {r: [] for r in robots}
    remaining_tasks = copy.deepcopy(tasks)
    
    # Get robot parameters
    battery_capacity = params.get('battery_capacity', float('inf')) if params else float('inf')
    max_distance = params.get('max_distance', float('inf')) if params else float('inf')
    recharge_time = params.get('recharge_time', 0) if params else 0
    
    # Track battery and distance for each robot
    robot_battery = {r: battery_capacity for r in robots}
    robot_distance = {r: 0 for r in robots}
    # Track time for each robot (for scheduling with recharge time)
    robot_time = {r: 0 for r in robots}
    
    # Use balance factor if available
    balance_factor = params.get('balance_factor', 1.0) if params else 1.0
    
    while remaining_tasks:
        # Check if any robots need recharging
        for robot in robots:
            # Calculate percentage battery remaining
            battery_percentage = robot_battery[robot] / battery_capacity if battery_capacity != float('inf') else 1.0
            
            # Recharge if battery is below 20%
            if battery_capacity != float('inf') and battery_percentage <= 0.2:
                logger.info(f"Robot {robot} battery at {battery_percentage:.2%}, needs recharging")
                # Get the robot's charging station
                station = get_robot_station(robot)
                if station in graph.nodes:
                    try:
                        # Path to charging station
                        path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                        station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                        
                        # Update cumulative distance
                        robot_distance[robot] += station_cost
                        
                        logger.info(f"Robot {robot} traveling to station {station} at cost {station_cost:.2f}, will recharge for {recharge_time} minutes")
                        # Add recharging path to assignment
                        assignments[robot].append({
                            'task': 'recharge',
                            'path': path_to_station,
                            'battery_remaining': battery_capacity,
                            'total_distance': robot_distance[robot],
                            'recharge_time': recharge_time
                        })
                        
                        # Update position, distance and time
                        robot_locations[robot] = station
                        robot_distance[robot] = 0  # Reset distance for battery calc
                        robot_battery[robot] = battery_capacity
                        # Add recharge time to robot's schedule
                        robot_time[robot] += station_cost + recharge_time
                        logger.info(f"Robot {robot} recharged to {robot_battery[robot]:.2f}, total distance: {robot_distance[robot]:.2f}")
                    except nx.NetworkXNoPath:
                        # If no path to station, just recharge in place
                        robot_battery[robot] = battery_capacity
                        # Still add recharge time
                        robot_time[robot] += recharge_time
                        assignments[robot].append({
                            'task': 'recharge_in_place',
                            'battery_remaining': battery_capacity,
                            'recharge_time': recharge_time
                        })
                        logger.info(f"Robot {robot} recharged in place, no path to station found")
                else:
                    # If station not in graph, recharge in place
                    robot_battery[robot] = battery_capacity
                    # Still add recharge time
                    robot_time[robot] += recharge_time
                    assignments[robot].append({
                        'task': 'recharge_in_place',
                        'battery_remaining': battery_capacity,
                        'recharge_time': recharge_time
                    })
                    logger.info(f"Robot {robot} recharged in place, station {station} not in graph")

        n_robots = len(robots)
        n_tasks = len(remaining_tasks)
        n = min(n_robots, n_tasks)
        cost_matrix = np.zeros((n_robots, n_tasks))
        # Pad cost matrix if robots > tasks or tasks > robots
        for i, robot in enumerate(robots):
            for j, (pickup, dropoff) in enumerate(remaining_tasks):
                try:
                    # Use compute_cost with parameters
                    logger.info(f"Computing cost for robot {robot} to task {pickup} to {dropoff}")
                    base_cost = compute_cost(graph, robot_locations[robot], pickup, dropoff, params)
                    logger.info(f"Base cost: {base_cost}")
                    # Add robot's current time to the cost to account for scheduling
                    base_cost += robot_time[robot]
                    
                    # Add penalties for battery and distance constraints
                    if battery_capacity != float('inf') and robot_battery[robot] < base_cost:
                        logger.info(f"Battery constraint violated for robot {robot}")
                        # Consider cost of going to charging station first
                        station = get_robot_station(robot)
                        if station in graph.nodes:
                            try:
                                logger.info(f"Station found: {station}")
                                station_path = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                                station_cost = nx.path_weight(graph, station_path, weight='weight')
                                logger.info(f"Station cost: {station_cost}")
                                pickup_from_station = nx.shortest_path(graph, station, pickup, weight='weight')
                                pickup_cost = nx.path_weight(graph, pickup_from_station, weight='weight')
                                logger.info(f"Pickup cost: {pickup_cost}")
                                dropoff_cost = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                                logger.info(f"Dropoff cost: {dropoff_cost}")
                                
                                # New cost includes: current → station → pickup → dropoff + recharge time
                                base_cost = robot_time[robot] + station_cost + recharge_time + pickup_cost + dropoff_cost
                                logger.info(f"New base cost: {base_cost}")
                            except nx.NetworkXNoPath:
                                base_cost = 1e6  # High penalty if no path
                        else:
                            # If no charging station, apply high penalty
                            base_cost = 1e6
                            
                    if max_distance != float('inf') and robot_distance[robot] + base_cost > max_distance:
                        logger.info(f"Distance constraint violated for robot {robot}")
                        base_cost = 1e6  # High penalty if exceeds max distance
                        
                    # Add workload balancing factor
                    workload_penalty = len(assignments[robot]) * balance_factor
                    cost_matrix[i, j] = base_cost + workload_penalty
                    
                except nx.NetworkXNoPath:
                    cost_matrix[i, j] = 1e6

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_robots = set()
        assigned_tasks = set()
        for i, j in zip(row_ind, col_ind):
            logger.info(f"Assigning robot {i} to task {j}")
            if i < n_robots and j < n_tasks and cost_matrix[i, j] < 1e6:
                robot = robots[i]
                task = remaining_tasks[j]
                pickup, dropoff = task
                
                # Check if robot needs recharging first
                needs_recharge = False
                if battery_capacity != float('inf'):
                    path1_cost = nx.shortest_path_length(graph, robot_locations[robot], pickup, weight='weight')
                    path2_cost = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                    total_path_cost = path1_cost + path2_cost
                    
                    # Calculate battery needed for the path as a percentage of max_distance
                    battery_needed = total_path_cost / max_distance * battery_capacity if max_distance != float('inf') else 0
                    
                    if robot_battery[robot] < battery_needed:
                        needs_recharge = True
                        # Get the robot's charging station
                        station = get_robot_station(robot)
                        if station in graph.nodes:
                            try:
                                # Path to charging station
                                path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                                station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                                
                                # Add recharging path to assignment
                                assignments[robot].append({
                                    'task': 'recharge',
                                    'path': path_to_station,
                                    'battery_remaining': battery_capacity,
                                    'total_distance': robot_distance[robot],
                                    'recharge_time': recharge_time
                                })
                                
                                # Update position and battery
                                robot_locations[robot] = station
                                robot_distance[robot] = 0  # Reset distance traveled counter
                                robot_battery[robot] = battery_capacity
                                # Add recharge time to robot's schedule
                                robot_time[robot] += station_cost + recharge_time
                            except nx.NetworkXNoPath:
                                # If no path to station, just recharge in place
                                robot_battery[robot] = battery_capacity
                                # Still add recharge time
                                robot_time[robot] += recharge_time
                                assignments[robot].append({
                                    'task': 'recharge_in_place',
                                    'battery_remaining': battery_capacity,
                                    'recharge_time': recharge_time
                                })
                        else:
                            # If station not in graph, recharge in place
                            robot_battery[robot] = battery_capacity
                            # Still add recharge time
                            robot_time[robot] += recharge_time
                            assignments[robot].append({
                                'task': 'recharge_in_place',
                                'battery_remaining': battery_capacity,
                                'recharge_time': recharge_time
                            })
                
                # Calculate path after potential recharging
                logger.info(f"Calculating path for robot {robot} to task {pickup} to {dropoff}")
                path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
                logger.info(f"Path 1: {path1}")
                path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
                logger.info(f"Path 2: {path2}")
                if path2[0] == pickup:
                    path2 = path2[1:]
                full_path = path1 + path2
                logger.info(f"Full path: {full_path}")
                # Calculate path cost
                path_cost = nx.path_weight(graph, full_path, weight='weight')
                logger.info(f"Path cost: {path_cost}")
                # Update robot distance
                robot_distance[robot] += path_cost
                logger.info(f"Robot distance: {robot_distance[robot]}")
                # Update battery using calculate_battery_remaining function
                logger.info(f"Battery capacity: {battery_capacity}")
                logger.info(f"Max distance: {max_distance}")
                robot_battery[robot] = calculate_battery_remaining(
                    battery_capacity, 
                    max_distance,
                    robot_distance[robot]
                )
                logger.info(f"Robot battery: {robot_battery[robot]}")
                logger.info(f"Robot {robot} assigned task {task}, new battery level: {robot_battery[robot]:.2f}%, distance: {robot_distance[robot]:.2f}")
                
                # Update robot time (for scheduling)
                robot_time[robot] += path_cost
                
                # Update assignment with cumulative distance
                assignments[robot].append({
                    'task': task, 
                    'path': full_path,
                    'battery_remaining': robot_battery[robot],
                    'total_distance': robot_distance[robot],
                    'completion_time': robot_time[robot]
                })
                robot_locations[robot] = dropoff
                assigned_robots.add(i)
                assigned_tasks.add(j)
        
        # Remove assigned tasks
        new_remaining = []
        for idx, t in enumerate(remaining_tasks):
            if idx not in assigned_tasks:
                new_remaining.append(t)
        remaining_tasks = new_remaining
    
    return assignments

def kmeans_clustering_assignment(tasks, robots, graph, env_json, params=None, n_clusters=None):
    """K-Means Inspired Task Clustering using JSON for coordinates."""
    if n_clusters is None:
        n_clusters = len(robots)
    
    logger.info(f"Starting kmeans_clustering_assignment with {len(tasks)} tasks, {len(robots)} robots, {n_clusters} clusters")
    logger.info(f"Parameters: battery_capacity={params.get('battery_capacity', 'inf')}, max_distance={params.get('max_distance', 'inf')}")
    
    # Build a mapping from node id to (x, y)
    coord_map = {node['id']: (node['x'], node['y']) for node in env_json['nodes'] if 'x' in node and 'y' in node}
    coords = np.array([coord_map[p] for p, _ in tasks])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    labels = kmeans.labels_
    assignment = {r: [] for r in robots}
    
    # Get robot parameters
    battery_capacity = params.get('battery_capacity', float('inf')) if params else float('inf')
    max_distance = params.get('max_distance', float('inf')) if params else float('inf')
    recharge_time = params.get('recharge_time', 0) if params else 0
    
    # Track battery and distance for each robot
    robot_battery = {r: battery_capacity for r in robots}
    robot_distance = {r: 0 for r in robots}
    # Track time for each robot (for scheduling with recharge time)
    robot_time = {r: 0 for r in robots}
    
    # Helper: greedy task ordering per robot
    def order_tasks(robot, task_list):
        ordered = []
        pos = robot
        remaining = task_list[:]
        while remaining:
            # pick task with min cost from current pos
            next_task = min(remaining, key=lambda t: compute_cost(graph, pos, t[0], t[1], params))
            ordered.append(next_task)
            pos = next_task[1]
            remaining.remove(next_task)
        return ordered

    for idx, robot in enumerate(robots):
        cluster_tasks = [tasks[i] for i, lb in enumerate(labels) if lb == idx]
        if not cluster_tasks:
            continue
        ordered = order_tasks(robot, cluster_tasks)
        current_pos = robot
        for pickup, dropoff in ordered:
            # Check if robot needs recharging before this task
            if battery_capacity != float('inf'):
                try:
                    path_to_pickup_cost = nx.shortest_path_length(graph, current_pos, pickup, weight='weight')
                    pickup_to_dropoff_cost = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                    total_cost = path_to_pickup_cost + pickup_to_dropoff_cost
                    
                    if robot_battery[robot] < total_cost:
                        # Get the robot's charging station
                        station = get_robot_station(robot)
                        if station in graph.nodes:
                            # Path to charging station
                            path_to_station = nx.shortest_path(graph, current_pos, station, weight='weight')
                            station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                            
                            # Add recharging path to assignment
                            assignment[robot].append({
                                'task': 'recharge',
                                'path': path_to_station,
                                'battery_remaining': battery_capacity,
                                'total_distance': robot_distance[robot],
                                'recharge_time': recharge_time
                            })
                            
                            # Update position and battery
                            current_pos = station
                            robot_distance[robot] = 0  # Reset distance traveled counter
                            robot_battery[robot] = battery_capacity
                            # Add recharge time to robot's schedule
                            robot_time[robot] += station_cost + recharge_time
                        else:
                            # If station not in graph, recharge in place
                            robot_battery[robot] = battery_capacity
                            # Still add recharge time
                            robot_time[robot] += recharge_time
                            assignment[robot].append({
                                'task': 'recharge_in_place',
                                'battery_remaining': battery_capacity,
                                'recharge_time': recharge_time
                            })
                except nx.NetworkXNoPath:
                    # If no path calculation is possible, just recharge in place
                    robot_battery[robot] = battery_capacity
                    # Still add recharge time
                    robot_time[robot] += recharge_time
                    assignment[robot].append({
                        'task': 'recharge_in_place',
                        'battery_remaining': battery_capacity,
                        'recharge_time': recharge_time
                    })
            
            # Calculate path after potential recharging
            path1 = nx.shortest_path(graph, current_pos, pickup, weight='weight')
            path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
            full = path1 + path2[1:]
            
            # Calculate path cost
            path_cost = nx.path_weight(graph, full, weight='weight')
            
            # Update battery and distance
            robot_distance[robot] += path_cost
            
            # Update battery using the calculation function
            robot_battery[robot] = calculate_battery_remaining(
                battery_capacity, 
                max_distance,
                robot_distance[robot]
            )
            logger.info(f"Robot {robot} completed task ({pickup}, {dropoff}), battery: {robot_battery[robot]:.2f}%, distance: {robot_distance[robot]:.2f}")
            
            # Update robot time (for scheduling)
            robot_time[robot] += path_cost
            
            assignment[robot].append({
                'task': (pickup, dropoff), 
                'path': full,
                'battery_remaining': robot_battery[robot],
                'total_distance': robot_distance[robot],
                'completion_time': robot_time[robot]
            })
            current_pos = dropoff
    return assignment

def auction_based_assignment(tasks, robots, graph, env_json, params=None):
    """
    Assign tasks via a simple auction: for each unassigned task, all robots 'bid' (distance cost), 
    and task is assigned to the robot with the lowest cost for it. Robots are updated after each win.
    """
    import copy
    
    logger.info(f"Starting auction_based_assignment with {len(tasks)} tasks and {len(robots)} robots")
    logger.info(f"Parameters: battery_capacity={params.get('battery_capacity', 'inf')}, max_distance={params.get('max_distance', 'inf')}")
    
    robot_locations = {r: r for r in robots}
    assignments = {r: [] for r in robots}
    pending_tasks = copy.deepcopy(tasks)
    
    # Get robot parameters
    battery_capacity = params.get('battery_capacity', float('inf')) if params else float('inf')
    max_distance = params.get('max_distance', float('inf')) if params else float('inf')
    recharge_time = params.get('recharge_time', 0) if params else 0
    
    # Track battery and distance for each robot
    robot_battery = {r: battery_capacity for r in robots}
    robot_distance = {r: 0 for r in robots}
    # Add cumulative distance tracker
    # cumulative_distance = {r: 0 for r in robots}
    # Track time for each robot (for scheduling with recharge time)
    robot_time = {r: 0 for r in robots}
    
    # Get objective weights if available
    objective_weights = params.get('objective_weights', {}) if params else {}
    distance_weight = objective_weights.get('distance', 1.0)
    balance_weight = objective_weights.get('balance', 0.5)
    
    while pending_tasks:
        # Check if any robots need recharging first
        for robot in robots:
            if battery_capacity != float('inf') and robot_battery[robot] <= 0:
                # Get the robot's charging station
                station = get_robot_station(robot)
                if station in graph.nodes:
                    try:
                        # Path to charging station
                        path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                        station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                        
                        # Add recharging path to assignment
                        assignments[robot].append({
                            'task': 'recharge',
                            'path': path_to_station,
                            'battery_remaining': battery_capacity,
                            'total_distance': robot_distance[robot] + station_cost,
                            'recharge_time': recharge_time
                        })
                        
                        # Update position and battery
                        robot_locations[robot] = station
                        robot_distance[robot] = 0  # Reset distance traveled counter
                        robot_battery[robot] = battery_capacity
                        # Add recharge time to robot's schedule
                        robot_time[robot] += station_cost + recharge_time
                    except nx.NetworkXNoPath:
                        # If no path to station, just recharge in place
                        robot_battery[robot] = battery_capacity
                        # Still add recharge time
                        robot_time[robot] += recharge_time
                        assignments[robot].append({
                            'task': 'recharge_in_place',
                            'battery_remaining': battery_capacity,
                            'recharge_time': recharge_time
                        })
                else:
                    # If station not in graph, recharge in place
                    robot_battery[robot] = battery_capacity
                    # Still add recharge time
                    robot_time[robot] += recharge_time
                    assignments[robot].append({
                        'task': 'recharge_in_place',
                        'battery_remaining': battery_capacity,
                        'recharge_time': recharge_time
                    })
        
        bids = []
        for task in pending_tasks:
            pickup, dropoff = task
            for robot in robots:
                loc = robot_locations[robot]
                try:
                    # Calculate base cost
                    base_cost = compute_cost(graph, loc, pickup, dropoff, params)
                    
                    # Include robot's current time in the cost for scheduling
                    base_cost += robot_time[robot]
                    
                    # Check if robot needs to recharge for this task
                    if battery_capacity != float('inf') and robot_battery[robot] < base_cost:
                        # Calculate cost with recharging stop
                        station = get_robot_station(robot)
                        if station in graph.nodes:
                            try:
                                # Calculate full path: current → station → pickup → dropoff + recharge time
                                station_path_cost = nx.shortest_path_length(graph, loc, station, weight='weight')
                                station_to_pickup_cost = nx.shortest_path_length(graph, station, pickup, weight='weight')
                                total_cost = robot_time[robot] + station_path_cost + recharge_time + station_to_pickup_cost + nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                                
                                # Apply objective weights
                                weighted_cost = total_cost * distance_weight
                                
                                # Add workload balance factor
                                if balance_weight > 0:
                                    workload_penalty = len(assignments[robot]) * balance_weight
                                    weighted_cost += workload_penalty
                                
                                # Include recharging in the bid (marked with needs_recharge flag)
                                bids.append((weighted_cost, robot, task, True))
                                continue
                            except nx.NetworkXNoPath:
                                continue
                        else:
                            continue
                    
                    # Check distance constraints
                    if max_distance != float('inf') and robot_distance[robot] + base_cost > max_distance:
                        continue
                    
                    # Apply objective weights
                    weighted_cost = base_cost * distance_weight
                    
                    # Add workload balance factor
                    if balance_weight > 0:
                        workload_penalty = len(assignments[robot]) * balance_weight
                        weighted_cost += workload_penalty
                    
                    bids.append((weighted_cost, robot, task, False))
                except nx.NetworkXNoPath:
                    continue
        
        if not bids:
            # If no valid bids, try recharging robots again
            any_recharged = False
            for robot in robots:
                if robot_battery[robot] < battery_capacity:
                    station = get_robot_station(robot)
                    if station in graph.nodes:
                        try:
                            path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                            station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                            
                            assignments[robot].append({
                                'task': 'recharge',
                                'path': path_to_station,
                                'battery_remaining': battery_capacity,
                                'total_distance': robot_distance[robot] + station_cost,
                                'recharge_time': recharge_time
                            })
                            
                            robot_locations[robot] = station
                            robot_distance[robot] = 0  # Reset distance traveled counter
                            robot_battery[robot] = battery_capacity
                            # Add recharge time to robot's schedule
                            robot_time[robot] += station_cost + recharge_time
                            any_recharged = True
                        except nx.NetworkXNoPath:
                            robot_battery[robot] = battery_capacity
                            # Still add recharge time
                            robot_time[robot] += recharge_time
                            assignments[robot].append({
                                'task': 'recharge_in_place',
                                'battery_remaining': battery_capacity,
                                'recharge_time': recharge_time
                            })
                            any_recharged = True
            
            if not any_recharged:
                break  # If no robot recharged, we can't proceed
                
        else:
            # Pick the task-robot pair with lowest cost
            best_bid = min(bids, key=lambda x: x[0])
            cost, robot, task, needs_recharge = best_bid
            pickup, dropoff = task
            
            # Handle recharging if needed
            if needs_recharge:
                station = get_robot_station(robot)
                path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                
                assignments[robot].append({
                    'task': 'recharge',
                    'path': path_to_station,
                    'battery_remaining': battery_capacity,
                    'total_distance': robot_distance[robot],
                    'recharge_time': recharge_time
                })
                
                robot_locations[robot] = station
                robot_distance[robot] = 0  # Reset distance traveled counter
                robot_battery[robot] = battery_capacity
                # Add recharge time to robot's schedule
                robot_time[robot] += station_cost + recharge_time
            
            # Now calculate the path to the pickup/dropoff
            path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
            path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
            if path2[0] == pickup:
                path2 = path2[1:]
            full_path = path1 + path2
            
            # Calculate path cost
            path_cost = nx.path_weight(graph, full_path, weight='weight')
            
            # Update both distance trackers
            robot_distance[robot] += path_cost
            # cumulative_distance[robot] += path_cost
            
            # Update battery using calculate_battery_remaining function
            robot_battery[robot] = calculate_battery_remaining(
                battery_capacity, 
                max_distance,
                robot_distance[robot]
            )
            logger.info(f"Robot {robot} won auction for task {task}, battery: {robot_battery[robot]:.2f}%, distance: {robot_distance[robot]:.2f}")
            
            # Update robot time (for scheduling)
            robot_time[robot] += path_cost
            
            # Update assignment with cumulative distance
            assignments[robot].append({
                'task': task, 
                'path': full_path,
                'battery_remaining': robot_battery[robot],
                'total_distance': robot_distance[robot],
                'completion_time': robot_time[robot]
            })
            robot_locations[robot] = dropoff
            pending_tasks.remove(task)
    
    return assignments

def genetic_algorithm_assignment(tasks, robots, graph, env_json, params=None, population_size=50, generations=100, mutation_rate=0.1):
    """5. Genetic Algorithm for Assignment."""
    
    logger.info(f"Starting genetic_algorithm_assignment with {len(tasks)} tasks and {len(robots)} robots")
    logger.info(f"GA Parameters: pop_size={population_size}, generations={generations}, mutation_rate={mutation_rate}")
    logger.info(f"Battery Parameters: battery_capacity={params.get('battery_capacity', 'inf')}, max_distance={params.get('max_distance', 'inf')}")
    
    n_tasks = len(tasks)
    n_robots = len(robots)
    
    # Get robot parameters
    battery_capacity = params.get('battery_capacity', float('inf')) if params else float('inf')
    max_distance = params.get('max_distance', float('inf')) if params else float('inf')
    recharge_time = params.get('recharge_time', 0) if params else 0
    
    # Get objective weights if available
    objective_weights = params.get('objective_weights', {}) if params else {}
    distance_weight = objective_weights.get('distance', 1.0)
    balance_weight = objective_weights.get('balance', 0.5)
    
    # Override algorithm parameters if provided
    if params:
        if 'population_size' in params:
            population_size = params['population_size']
        if 'generations' in params:
            generations = params['generations']
        if 'mutation_rate' in params:
            mutation_rate = params['mutation_rate']
    
    # chromosome: list of length n_tasks, genes in [0, n_robots)
    def decode(chromo):
        assign = {r: [] for r in robots}
        pos = {r: r for r in robots}
        robot_battery = {r: battery_capacity for r in robots}
        robot_distance = {r: 0 for r in robots}
        robot_time = {r: 0 for r in robots}
        
        for task_idx, gene in enumerate(chromo):
            r = robots[gene]
            pickup, dropoff = tasks[task_idx]
            
            # Check if robot needs to recharge before this task
            if battery_capacity != float('inf'):
                try:
                    path_to_pickup_cost = nx.shortest_path_length(graph, pos[r], pickup, weight='weight')
                    pickup_to_dropoff_cost = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                    total_cost = path_to_pickup_cost + pickup_to_dropoff_cost
                    
                    if robot_battery[r] < total_cost:
                        # Get the robot's charging station
                        station = get_robot_station(r)
                        if station in graph.nodes:
                            # Path to charging station
                            path_to_station = nx.shortest_path(graph, pos[r], station, weight='weight')
                            station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                            
                            # Add recharging path to assignment
                            assign[r].append({
                                'task': 'recharge',
                                'path': path_to_station,
                                'battery_remaining': battery_capacity,
                                'total_distance': robot_distance[r],
                                'recharge_time': recharge_time
                            })
                            
                            # Update position and battery
                            pos[r] = station
                            robot_distance[r] = 0  # Reset distance traveled counter
                            robot_battery[r] = battery_capacity
                            # Add recharge time to robot's schedule
                            robot_time[r] += station_cost + recharge_time
                        else:
                            # If station not in graph, recharge in place
                            robot_battery[r] = battery_capacity
                            # Still add recharge time
                            robot_time[r] += recharge_time
                            assign[r].append({
                                'task': 'recharge_in_place',
                                'battery_remaining': battery_capacity,
                                'recharge_time': recharge_time
                            })
                except nx.NetworkXNoPath:
                    # If no path calculation is possible, just recharge in place
                    robot_battery[r] = battery_capacity
                    # Still add recharge time
                    robot_time[r] += recharge_time
                    assign[r].append({
                        'task': 'recharge_in_place',
                        'battery_remaining': battery_capacity,
                        'recharge_time': recharge_time
                    })
            
            # Calculate path after potential recharging
            try:
                path1 = nx.shortest_path(graph, pos[r], pickup, weight='weight')
                path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
                full = path1 + path2[1:]
                
                # Calculate path cost
                path_cost = nx.path_weight(graph, full, weight='weight')
                
                # Check if battery would be sufficient
                if battery_capacity != float('inf'):
                    battery_needed = path_cost / max_distance * battery_capacity
                    if robot_battery[r] < battery_needed:
                        # Skip this task if not enough battery
                        continue
                
                # Check if would exceed max distance
                if max_distance != float('inf'):
                    if robot_distance[r] + path_cost > max_distance:
                        # Skip if exceeds max distance
                        continue
                
                robot_distance[r] += path_cost
                
                # Update battery using calculate_battery_remaining function
                robot_battery[r] = calculate_battery_remaining(
                    battery_capacity, 
                    max_distance,
                    robot_distance[r]
                )
                
                # Update robot time (for scheduling)
                robot_time[r] += path_cost
                
                assign[r].append({
                    'task': (pickup, dropoff), 
                    'path': full,
                    'battery_remaining': robot_battery[r],
                    'total_distance': robot_distance[r],
                    'completion_time': robot_time[r]
                })
            except nx.NetworkXNoPath:
                # Skip this task if no path found
                continue
            
        return assign

    def fitness(chromo):
        assign = decode(chromo)
        # total distance with weights
        total_dist = sum(
            sum(nx.path_weight(graph, entry['path'], weight='weight') for entry in assign[r] if entry['task'] != 'recharge')
            for r in robots
        ) * distance_weight
        
        # Count only actual tasks (not recharge operations)
        actual_tasks = {r: [entry for entry in assign[r] if entry['task'] != 'recharge']
                         for r in robots}
        
        # Calculate time fitness (maximum completion time across all robots)
        completion_times = [entry.get('completion_time', 0) 
                           for r in robots 
                           for entry in assign[r] 
                           if entry.get('task') != 'recharge' and entry.get('task') != 'recharge_in_place']
        
        makespan = max(completion_times) if completion_times else 0
        
        # balance penalty = std dev of task counts
        counts = np.array([len(actual_tasks[r]) for r in robots], dtype=float)
        balance_penalty = counts.std() * balance_weight
        
        # Failed assignment penalty (tasks that couldn't be assigned due to constraints)
        assigned_task_count = sum(len(actual_tasks[r]) for r in robots)
        failed_penalty = (n_tasks - assigned_task_count) * 1000  # High penalty for unassigned tasks
        
        return total_dist + balance_penalty + failed_penalty + makespan * 0.5

    # initialize population
    population = [np.random.randint(0, n_robots, size=n_tasks).tolist() for _ in range(population_size)]
    for _ in range(generations):
        # evaluate
        scores = [(fitness(chromo), chromo) for chromo in population]
        scores.sort(key=lambda x: x[0])
        # select top 20%
        cutoff = max(2, int(population_size * 0.2))
        parents = [chromo for _, chromo in scores[:cutoff]]
        # generate new population
        new_pop = parents[:]
        while len(new_pop) < population_size:
            p1, p2 = random.sample(parents, 2)
            # single-point crossover
            point = random.randrange(1, n_tasks)
            child = p1[:point] + p2[point:]
            # mutation
            for idx in range(n_tasks):
                if random.random() < mutation_rate:
                    child[idx] = random.randrange(0, n_robots)
            new_pop.append(child)
        population = new_pop

    # best chromosome
    best = min(population, key=lambda c: fitness(c))
    return decode(best)

def makespan_balanced_assignment(tasks, robots, graph, env_json, params=None):
    """
    Assign each task to the robot with the current smallest total path length.
    """
    import copy
    
    logger.info(f"Starting makespan_balanced_assignment with {len(tasks)} tasks and {len(robots)} robots")
    logger.info(f"Parameters: battery_capacity={params.get('battery_capacity', 'inf')}, max_distance={params.get('max_distance', 'inf')}")
    
    robot_locations = {r: r for r in robots}
    assignments = {r: [] for r in robots}
    total_dists = {r: 0 for r in robots}
    pending_tasks = copy.deepcopy(tasks)
    
    # Get robot parameters
    battery_capacity = params.get('battery_capacity', float('inf')) if params else float('inf')
    max_distance = params.get('max_distance', float('inf')) if params else float('inf')
    recharge_time = params.get('recharge_time', 0) if params else 0
    
    # Track battery and distance for each robot
    robot_battery = {r: battery_capacity for r in robots}
    robot_distance = {r: 0 for r in robots}
    # Track time for each robot (for scheduling with recharge time)
    robot_time = {r: 0 for r in robots}

    while pending_tasks:
        # Check for any robots that need recharging
        recharged_robots = False
        for robot in robots:
            if battery_capacity != float('inf') and robot_battery[robot] <= battery_capacity * 0.2:  # Recharge when at 20% battery
                # Get the robot's charging station
                station = get_robot_station(robot)
                if station in graph.nodes:
                    try:
                        # Path to charging station
                        path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                        station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                        
                        # Add recharging path to assignment
                        assignments[robot].append({
                            'task': 'recharge',
                            'path': path_to_station,
                            'battery_remaining': battery_capacity,
                            'total_distance': robot_distance[robot],
                            'recharge_time': recharge_time
                        })
                        
                        # Update position and battery
                        robot_locations[robot] = station
                        robot_distance[robot] = 0  # Reset distance traveled counter
                        robot_battery[robot] = battery_capacity
                        # Add recharge time to robot's schedule
                        robot_time[robot] += station_cost + recharge_time
                        recharged_robots = True
                    except nx.NetworkXNoPath:
                        # If no path to station, just recharge in place
                        robot_battery[robot] = battery_capacity
                        # Still add recharge time
                        robot_time[robot] += recharge_time
                        assignments[robot].append({
                            'task': 'recharge_in_place',
                            'battery_remaining': battery_capacity,
                            'recharge_time': recharge_time
                        })
                        recharged_robots = True
                else:
                    # If station not in graph, recharge in place
                    robot_battery[robot] = battery_capacity
                    # Still add recharge time
                    robot_time[robot] += recharge_time
                    assignments[robot].append({
                        'task': 'recharge_in_place',
                        'battery_remaining': battery_capacity,
                        'recharge_time': recharge_time
                    })
                    recharged_robots = True
        
        best = None
        for robot in robots:
            loc = robot_locations[robot]
            for task in pending_tasks:
                pickup, dropoff = task
                try:
                    # Calculate costs using robot speed if available
                    d1 = nx.shortest_path_length(graph, loc, pickup, weight='weight')
                    d2 = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                    
                    # Apply robot speed
                    if params and 'robot_speed' in params:
                        robot_speed = params.get('robot_speed', 1.0)
                        if robot_speed > 0:
                            d1 = d1 / robot_speed
                            d2 = d2 / robot_speed
                    
                    # Include current robot time in cost to account for scheduling
                    time_cost = robot_time[robot] + d1 + d2
                    
                    # Check if recharging is needed
                    needs_recharge = False
                    recharge_cost = 0
                    if battery_capacity != float('inf') and robot_battery[robot] < (d1 + d2):
                        # Calculate recharging costs
                        station = get_robot_station(robot)
                        if station in graph.nodes:
                            try:
                                # Calculate detour to charging station
                                station_cost = nx.shortest_path_length(graph, loc, station, weight='weight')
                                station_to_pickup = nx.shortest_path_length(graph, station, pickup, weight='weight')
                                recharge_cost = station_cost + station_to_pickup - d1  # Additional cost of detour
                                time_cost += recharge_cost + recharge_time  # Add recharge time
                                needs_recharge = True
                            except nx.NetworkXNoPath:
                                continue
                        else:
                            # If no station, just add recharge time
                            time_cost += recharge_time
                            needs_recharge = True
                    
                    # Check distance constraints
                    if max_distance != float('inf') and robot_distance[robot] + d1 + d2 + recharge_cost > max_distance:
                        continue
                    
                    # Use time cost for scheduling optimization
                    if (best is None) or (time_cost < best[0]):
                        best = (time_cost, robot, task, d1, d2, needs_recharge)
                except nx.NetworkXNoPath:
                    continue
        
        if best is None:
            # If no valid assignment and we haven't already recharged robots
            if not recharged_robots:
                for robot in robots:
                    if battery_capacity != float('inf'):
                        robot_battery[robot] = battery_capacity  # Just recharge in place as last resort
                        # Still add recharge time
                        robot_time[robot] += recharge_time
                        assignments[robot].append({
                            'task': 'recharge_in_place',
                            'battery_remaining': battery_capacity,
                            'recharge_time': recharge_time
                        })
                continue
            break  # No more assignments possible
        
        _, robot, task, d1, d2, needs_recharge = best
        pickup, dropoff = task
        
        # Handle recharging if needed
        if needs_recharge:
            station = get_robot_station(robot)
            if station in graph.nodes:
                try:
                    path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                    station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                    
                    # Add recharging path
                    assignments[robot].append({
                        'task': 'recharge',
                        'path': path_to_station,
                        'battery_remaining': battery_capacity,
                        'total_distance': robot_distance[robot],
                        'recharge_time': recharge_time
                    })
                    
                    # Update robot position, battery, and distance
                    robot_locations[robot] = station
                    robot_distance[robot] = 0  # Reset distance traveled counter
                    robot_battery[robot] = battery_capacity
                    # Add recharge time to robot's schedule
                    robot_time[robot] += station_cost + recharge_time
                    
                    # Recalculate path from station
                    path1 = nx.shortest_path(graph, station, pickup, weight='weight')
                except nx.NetworkXNoPath:
                    # If no path to station, recharge in place
                    robot_battery[robot] = battery_capacity
                    # Still add recharge time
                    robot_time[robot] += recharge_time
                    assignments[robot].append({
                        'task': 'recharge_in_place',
                        'battery_remaining': battery_capacity,
                        'recharge_time': recharge_time
                    })
                    # Calculate normal path
                    path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
            else:
                # If no station, recharge in place and calculate normal path
                robot_battery[robot] = battery_capacity
                # Still add recharge time
                robot_time[robot] += recharge_time
                assignments[robot].append({
                    'task': 'recharge_in_place',
                    'battery_remaining': battery_capacity,
                    'recharge_time': recharge_time
                })
                path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
        else:
            # Normal path calculation
            path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
        
        path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
        if path2[0] == pickup:
            path2 = path2[1:]
        full_path = path1 + path2
        
        # Calculate path cost
        path_cost = nx.path_weight(graph, full_path, weight='weight')
        
        # Update both distance trackers
        robot_distance[robot] += path_cost
        # cumulative_distance[robot] += path_cost
        
        # Update battery using calculate_battery_remaining function
        robot_battery[robot] = calculate_battery_remaining(
            battery_capacity, 
            max_distance,
            robot_distance[robot]
        )
        logger.info(f"Robot {robot} assigned to task {task} for best makespan, battery: {robot_battery[robot]:.2f}%, distance: {robot_distance[robot]:.2f}")
        
        # Update time (for scheduling)
        robot_time[robot] += path_cost
        
        assignments[robot].append({
            'task': task, 
            'path': full_path,
            'battery_remaining': robot_battery[robot],
            'total_distance': robot_distance[robot],
            'completion_time': robot_time[robot]
        })
        robot_locations[robot] = dropoff
        total_dists[robot] += d1 + d2
        pending_tasks.remove(task)
        
    return assignments

def marginal_benefit_assignment(tasks, robots, graph, env_json, params=None):
    """
    Assign task to robot whose total travel distance increases the least by adding the task.
    """
    import copy
    
    logger.info(f"Starting marginal_benefit_assignment with {len(tasks)} tasks and {len(robots)} robots")
    logger.info(f"Parameters: battery_capacity={params.get('battery_capacity', 'inf')}, max_distance={params.get('max_distance', 'inf')}")
    
    robot_locations = {r: r for r in robots}
    assignments = {r: [] for r in robots}
    total_dists = {r: 0 for r in robots}
    pending_tasks = copy.deepcopy(tasks)
    
    # Get robot parameters
    battery_capacity = params.get('battery_capacity', float('inf')) if params else float('inf')
    max_distance = params.get('max_distance', float('inf')) if params else float('inf')
    recharge_time = params.get('recharge_time', 0) if params else 0
    
    # Track battery and distance for each robot
    robot_battery = {r: battery_capacity for r in robots}
    robot_distance = {r: 0 for r in robots}
    # Track time for each robot (for scheduling with recharge time)
    robot_time = {r: 0 for r in robots}
    
    # Get balance factor if available
    balance_factor = params.get('balance_factor', 1.0) if params else 1.0

    while pending_tasks:
        # First check and handle any low battery robots
        for robot in robots:
            if battery_capacity != float('inf') and robot_battery[robot] <= battery_capacity * 0.2:  # Recharge at 20% battery
                # Get the robot's charging station
                station = get_robot_station(robot)
                if station in graph.nodes:
                    try:
                        # Path to charging station
                        path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                        station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                        
                        # Add recharging path to assignment
                        assignments[robot].append({
                            'task': 'recharge',
                            'path': path_to_station,
                            'battery_remaining': battery_capacity,
                            'total_distance': robot_distance[robot],
                            'recharge_time': recharge_time
                        })
                        
                        # Update position and battery
                        robot_locations[robot] = station
                        robot_distance[robot] = 0  # Reset distance traveled counter
                        robot_battery[robot] = battery_capacity
                        # Add recharge time to robot's schedule
                        robot_time[robot] += station_cost + recharge_time
                    except nx.NetworkXNoPath:
                        # If no path to station, just recharge in place
                        robot_battery[robot] = battery_capacity
                        # Still add recharge time
                        robot_time[robot] += recharge_time
                        assignments[robot].append({
                            'task': 'recharge_in_place',
                            'battery_remaining': battery_capacity,
                            'recharge_time': recharge_time
                        })
                else:
                    # If station not in graph, recharge in place
                    robot_battery[robot] = battery_capacity
                    # Still add recharge time
                    robot_time[robot] += recharge_time
                    assignments[robot].append({
                        'task': 'recharge_in_place',
                        'battery_remaining': battery_capacity,
                        'recharge_time': recharge_time
                    })

        best = None
        for task in pending_tasks:
            pickup, dropoff = task
            for robot in robots:
                loc = robot_locations[robot]
                try:
                    # Calculate base cost
                    d1 = nx.shortest_path_length(graph, loc, pickup, weight='weight')
                    d2 = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                    
                    # Apply robot speed
                    if params and 'robot_speed' in params:
                        robot_speed = params.get('robot_speed', 1.0)
                        if robot_speed > 0:
                            d1 = d1 / robot_speed
                            d2 = d2 / robot_speed
                    
                    # Include current robot time for scheduling
                    time_cost = robot_time[robot] + d1 + d2
                    
                    # Check if recharging is needed
                    needs_recharge = False
                    recharge_cost = 0
                    if battery_capacity != float('inf') and robot_battery[robot] < (d1 + d2):
                        # Calculate recharging costs
                        station = get_robot_station(robot)
                        if station in graph.nodes:
                            try:
                                # Calculate detour to charging station
                                station_cost = nx.shortest_path_length(graph, loc, station, weight='weight')
                                station_to_pickup = nx.shortest_path_length(graph, station, pickup, weight='weight')
                                recharge_cost = station_cost + station_to_pickup - d1  # Additional cost of detour
                                time_cost += recharge_cost + recharge_time  # Add recharge time
                                needs_recharge = True
                            except nx.NetworkXNoPath:
                                continue
                        else:
                            # If no station, just add recharge time
                            time_cost += recharge_time
                            needs_recharge = True
                    
                    # Check distance constraints
                    if max_distance != float('inf') and robot_distance[robot] + d1 + d2 + recharge_cost > max_distance:
                        continue
                    
                    # Calculate marginal cost with workload balancing
                    marginal_cost = d1 + d2 + recharge_cost
                    
                    # Apply balance factor
                    if balance_factor > 0:
                        workload_penalty = len(assignments[robot]) * balance_factor
                        marginal_cost += workload_penalty
                    
                    # Use time cost for tie-breaking
                    if (best is None) or (marginal_cost < best[0]) or (marginal_cost == best[0] and time_cost < best[6]):
                        best = (marginal_cost, robot, task, d1, d2, needs_recharge, time_cost)
                except nx.NetworkXNoPath:
                    continue
                    
        if best is None:
            # If no valid assignment, try recharging all robots
            recharged = False
            for robot in robots:
                if battery_capacity != float('inf') and robot_battery[robot] < battery_capacity:
                    station = get_robot_station(robot)
                    if station in graph.nodes:
                        try:
                            path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                            station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                            
                            assignments[robot].append({
                                'task': 'recharge',
                                'path': path_to_station,
                                'battery_remaining': battery_capacity,
                                'total_distance': robot_distance[robot],
                                'recharge_time': recharge_time
                            })
                            
                            robot_locations[robot] = station
                            robot_distance[robot] = 0  # Reset distance traveled counter
                            robot_battery[robot] = battery_capacity
                            # Add recharge time to robot's schedule
                            robot_time[robot] += station_cost + recharge_time
                            recharged = True
                        except nx.NetworkXNoPath:
                            robot_battery[robot] = battery_capacity
                            # Still add recharge time
                            robot_time[robot] += recharge_time
                            assignments[robot].append({
                                'task': 'recharge_in_place',
                                'battery_remaining': battery_capacity,
                                'recharge_time': recharge_time
                            })
                            recharged = True
            
            if not recharged:
                break  # No more possible assignments
            continue  # Try again after recharging
        
        marginal_cost, robot, task, d1, d2, needs_recharge, _ = best
        pickup, dropoff = task
        
        # Handle recharging if needed
        if needs_recharge:
            station = get_robot_station(robot)
            if station in graph.nodes:
                try:
                    path_to_station = nx.shortest_path(graph, robot_locations[robot], station, weight='weight')
                    station_cost = nx.path_weight(graph, path_to_station, weight='weight')
                    
                    # Add recharging path
                    assignments[robot].append({
                        'task': 'recharge',
                        'path': path_to_station,
                        'battery_remaining': battery_capacity,
                        'total_distance': robot_distance[robot],
                        'recharge_time': recharge_time
                    })
                    
                    # Update position and battery
                    robot_locations[robot] = station
                    robot_distance[robot] = 0  # Reset distance traveled counter
                    robot_battery[robot] = battery_capacity
                    # Add recharge time to robot's schedule
                    robot_time[robot] += station_cost + recharge_time
                    
                    # Calculate path from station
                    path1 = nx.shortest_path(graph, station, pickup, weight='weight')
                except nx.NetworkXNoPath:
                    # If no path to station, recharge in place
                    robot_battery[robot] = battery_capacity
                    # Still add recharge time
                    robot_time[robot] += recharge_time
                    assignments[robot].append({
                        'task': 'recharge_in_place',
                        'battery_remaining': battery_capacity,
                        'recharge_time': recharge_time
                    })
                    # Calculate normal path
                    path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
            else:
                # If no station, recharge in place
                robot_battery[robot] = battery_capacity
                # Still add recharge time
                robot_time[robot] += recharge_time
                assignments[robot].append({
                    'task': 'recharge_in_place',
                    'battery_remaining': battery_capacity,
                    'recharge_time': recharge_time
                })
                path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
        else:
            # Normal path calculation
            path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
        
        path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
        if path2[0] == pickup:
            path2 = path2[1:]
        full_path = path1 + path2
        
        # Calculate path cost
        path_cost = nx.path_weight(graph, full_path, weight='weight')
        
        # Update both distance trackers
        robot_distance[robot] += path_cost
        # cumulative_distance[robot] += path_cost
        
        # Update battery using calculate_battery_remaining function
        robot_battery[robot] = calculate_battery_remaining(
            battery_capacity, 
            max_distance,
            robot_distance[robot]
        )
        logger.info(f"Robot {robot} assigned task {task} with marginal cost {marginal_cost:.2f}, battery: {robot_battery[robot]:.2f}%, distance: {robot_distance[robot]:.2f}")
        
        # Update time (for scheduling)
        robot_time[robot] += path_cost
        
        assignments[robot].append({
            'task': task, 
            'path': full_path,
            'battery_remaining': robot_battery[robot],
            'total_distance': robot_distance[robot],
            'completion_time': robot_time[robot]
        })
        robot_locations[robot] = dropoff
        pending_tasks.remove(task)
        
    return assignments
