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
        
        try:
            # Import the algorithm module and function
            module = importlib.import_module(algorithm.module_path)
            algo_function = getattr(module, algorithm.function_name)
            robots = environment.elements.get("robots")
            robots_ids = [r.get("id") for r in robots]
            nx_graph = build_undirected_graph_from_json(environment.graph)
            # Execute the algorithm
            result = algo_function(
                tasks=task.details.get("tasks"),
                robots=robots_ids,
                graph=nx_graph,
                env_json=environment.graph
            )
            
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


def compute_cost(graph, start, pickup, dropoff):
    print(f"start: {start}, pickup: {pickup}, dropoff: {dropoff}")
    """Compute cost = distance(start→pickup) + distance(pickup→dropoff)."""
    cost1 = nx.shortest_path_length(graph, start, pickup, weight='weight')
    cost2 = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
    return cost1 + cost2

def greedy_assignment(tasks, robots, graph, env_json):
    """1. Greedy Distance Minimization."""
    assignment = {r: [] for r in robots}
    robots_positions = {r: r for r in robots}
    print(f"assignment: {assignment}")
    print(f"graph: {graph}")
    for pickup, dropoff in tasks:
        # find robot with minimum cost
        print(f"pickup: {pickup}, dropoff: {dropoff}")
        best_robot = min(robots, key=lambda r: compute_cost(graph, robots_positions[r], pickup, dropoff))
        print(f"best_robot: {best_robot}")
        # build path
        path_to_pickup = nx.shortest_path(graph, robots_positions[best_robot], pickup, weight='weight')
        path_to_dropoff = nx.shortest_path(graph, pickup, dropoff, weight='weight')
        full_path = path_to_pickup + path_to_dropoff[1:]
        assignment[best_robot].append({'task': (pickup, dropoff), 'path': full_path})
        # update robot's new position for next iteration
        robots_positions[best_robot] = dropoff
    return assignment

def hungarian_assignment(tasks, robots, graph, env_json):
    """
    Assign all tasks among robots using the Hungarian Algorithm in batches.
    Each robot gets one task per batch, and their location is updated after each batch.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import copy

    robot_locations = {r: r for r in robots}
    assignments = {r: [] for r in robots}
    remaining_tasks = copy.deepcopy(tasks)
    
    while remaining_tasks:
        n_robots = len(robots)
        n_tasks = len(remaining_tasks)
        n = min(n_robots, n_tasks)
        cost_matrix = np.zeros((n_robots, n_tasks))
        # Pad cost matrix if robots > tasks or tasks > robots
        for i, robot in enumerate(robots):
            for j, (pickup, dropoff) in enumerate(remaining_tasks):
                try:
                    d1 = nx.shortest_path_length(graph, robot_locations[robot], pickup, weight='weight')
                    d2 = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                    cost_matrix[i, j] = d1 + d2
                except nx.NetworkXNoPath:
                    cost_matrix[i, j] = 1e6

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_robots = set()
        assigned_tasks = set()
        for i, j in zip(row_ind, col_ind):
            if i < n_robots and j < n_tasks and cost_matrix[i, j] < 1e6:
                robot = robots[i]
                task = remaining_tasks[j]
                pickup, dropoff = task
                path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
                path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
                if path2[0] == pickup:
                    path2 = path2[1:]
                assignments[robot].append({'task': task, 'path': path1 + path2})
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

def kmeans_clustering_assignment(tasks, robots, graph, env_json, n_clusters=None):
    """K-Means Inspired Task Clustering using JSON for coordinates."""
    if n_clusters is None:
        n_clusters = len(robots)
    # Build a mapping from node id to (x, y)
    coord_map = {node['id']: (node['x'], node['y']) for node in env_json['nodes'] if 'x' in node and 'y' in node}
    coords = np.array([coord_map[p] for p, _ in tasks])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    labels = kmeans.labels_
    assignment = {r: [] for r in robots}
    # Helper: greedy task ordering per robot
    def order_tasks(robot, task_list):
        ordered = []
        pos = robot
        remaining = task_list[:]
        while remaining:
            # pick task with min cost from current pos
            next_task = min(remaining, key=lambda t: compute_cost(graph, pos, t[0], t[1]))
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
            p1 = nx.shortest_path(graph, current_pos, pickup, weight='weight')
            p2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
            full = p1 + p2[1:]
            assignment[robot].append({'task': (pickup, dropoff), 'path': full})
            current_pos = dropoff
    return assignment

def auction_based_assignment(tasks, robots, graph, env_json):
    """
    Assign tasks via a simple auction: for each unassigned task, all robots 'bid' (distance cost), 
    and task is assigned to the robot with the lowest cost for it. Robots are updated after each win.
    """
    import copy
    robot_locations = {r: r for r in robots}
    assignments = {r: [] for r in robots}
    pending_tasks = copy.deepcopy(tasks)
    
    while pending_tasks:
        bids = []
        for task in pending_tasks:
            pickup, dropoff = task
            for robot in robots:
                loc = robot_locations[robot]
                try:
                    d1 = nx.shortest_path_length(graph, loc, pickup, weight='weight')
                    d2 = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                except nx.NetworkXNoPath:
                    continue
                cost = d1 + d2
                bids.append((cost, robot, task))
        if not bids:
            break
        # Pick the task-robot pair with lowest cost
        cost, robot, task = min(bids, key=lambda x: x[0])
        pickup, dropoff = task
        path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
        path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
        if path2[0] == pickup:
            path2 = path2[1:]
        assignments[robot].append({'task': task, 'path': path1 + path2})
        robot_locations[robot] = dropoff
        pending_tasks.remove(task)
    return assignments


def genetic_algorithm_assignment(tasks, robots, graph, env_json, population_size=50, generations=100, mutation_rate=0.1):
    """5. Genetic Algorithm for Assignment."""
    n_tasks = len(tasks)
    n_robots = len(robots)
    # chromosome: list of length n_tasks, genes in [0, n_robots)
    def decode(chromo):
        assign = {r: [] for r in robots}
        pos = {r: r for r in robots}
        for task_idx, gene in enumerate(chromo):
            r = robots[gene]
            pickup, dropoff = tasks[task_idx]
            path1 = nx.shortest_path(graph, pos[r], pickup, weight='weight')
            path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
            full = path1 + path2[1:]
            assign[r].append({'task': (pickup, dropoff), 'path': full})
            pos[r] = dropoff
        return assign

    def fitness(chromo):
        assign = decode(chromo)
        # total distance
        total_dist = sum(
            sum(nx.path_weight(graph, entry['path'], weight='weight') for entry in assign[r])
            for r in robots
        )
        # balance penalty = std dev of task counts
        counts = np.array([len(assign[r]) for r in robots], dtype=float)
        balance_penalty = counts.std()
        return total_dist + balance_penalty * 10  # weighted sum

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

def makespan_balanced_assignment(tasks, robots, graph, env_json):
    """
    Assign each task to the robot with the current smallest total path length.
    """
    import copy
    robot_locations = {r: r for r in robots}
    assignments = {r: [] for r in robots}
    total_dists = {r: 0 for r in robots}
    pending_tasks = copy.deepcopy(tasks)

    while pending_tasks:
        best = None
        for robot in robots:
            loc = robot_locations[robot]
            for task in pending_tasks:
                pickup, dropoff = task
                try:
                    d1 = nx.shortest_path_length(graph, loc, pickup, weight='weight')
                    d2 = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                except nx.NetworkXNoPath:
                    continue
                cost = total_dists[robot] + d1 + d2
                if (best is None) or (cost < best[0]):
                    best = (cost, robot, task, d1, d2)
        if best is None:
            break
        _, robot, task, d1, d2 = best
        pickup, dropoff = task
        path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
        path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
        if path2[0] == pickup:
            path2 = path2[1:]
        assignments[robot].append({'task': task, 'path': path1 + path2})
        robot_locations[robot] = dropoff
        total_dists[robot] += d1 + d2
        pending_tasks.remove(task)
    return assignments

def marginal_benefit_assignment(tasks, robots, graph, env_json):
    """
    Assign task to robot whose total travel distance increases the least by adding the task.
    """
    import copy
    robot_locations = {r: r for r in robots}
    assignments = {r: [] for r in robots}
    total_dists = {r: 0 for r in robots}
    pending_tasks = copy.deepcopy(tasks)

    while pending_tasks:
        best = None
        for task in pending_tasks:
            pickup, dropoff = task
            for robot in robots:
                loc = robot_locations[robot]
                try:
                    d1 = nx.shortest_path_length(graph, loc, pickup, weight='weight')
                    d2 = nx.shortest_path_length(graph, pickup, dropoff, weight='weight')
                except nx.NetworkXNoPath:
                    continue
                marginal_cost = d1 + d2
                if (best is None) or (marginal_cost < best[0]):
                    best = (marginal_cost, robot, task)
        if best is None:
            break
        _, robot, task = best
        pickup, dropoff = task
        path1 = nx.shortest_path(graph, robot_locations[robot], pickup, weight='weight')
        path2 = nx.shortest_path(graph, pickup, dropoff, weight='weight')
        if path2[0] == pickup:
            path2 = path2[1:]
        assignments[robot].append({'task': task, 'path': path1 + path2})
        robot_locations[robot] = dropoff
        pending_tasks.remove(task)
    return assignments
