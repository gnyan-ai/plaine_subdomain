import networkx as nx
import itertools
import random
import simpy
import zlib
from simpy.events import AnyOf, AllOf, Event, Process
import dill as pickle
from dill import dumps, loads
import pandas as pd
import numpy as np
import io


def init_part_data(bom_df):
    # Create a dictionary to map part names to lt_offset and safety_time values
    part_data = {}
    for index, row in bom_df.iterrows():
        # Assuming 'part' is the column containing part names
        part_name = row['part']
        part_data[part_name] = {
            'lt_offset': row['lt_offset']*7/5.0,
            'safety_time': row['safety_time']*7/5.0,
            'goods_receipt_time': row['gr_mday']*7/5.0,
            'planned_delivery_time': row['pdt_calday'],
            'schedule_margin_time': row['smk_mday']*7/5.0,
        }
    return part_data

def find_longest_path_with_a_star(graph, end_node, heuristic_distances):
    # Define a custom A* heuristic function that returns the heuristic distance for a node
    def heuristic(node, target):
        return -heuristic_distances[target]

    # Use the A* algorithm to find the longest path
    longest_path = nx.astar_path(
        graph, source=list(nx.topological_sort(graph))[0], target=end_node, heuristic=heuristic)

    return str(longest_path)
    
def collect_last_nodes_per_level_reverse(graph, end_node, completion_times):
    # Create a dictionary to store the last node per level
    last_nodes_per_level = {}

    # Initialize the BFS queue with the end node
    queue = [(end_node, 0)]  # Tuple: (node, level)
    keep_queue = [(end_node, 0)]
    while queue:
        node, level = queue.pop(0)  # Dequeue a node

        # Store the last node encountered at the current level
        last_nodes_per_level[level] = node

        # Enqueue parent nodes at the previous level
        for parent in graph.predecessors(node):
            queue.append((parent, level + 1))
            keep_queue.append((parent, level + 1))
            
    depth_df = pd.DataFrame(keep_queue, columns=['Part', 'Level'])
    depth_df['Time'] = depth_df.Part.apply(lambda x: completion_times[x])
    return depth_df.sort_values(by=['Level', 'Time'], ascending=[False, True]).drop_duplicates(subset=['Level'], keep='first')
    
def run_simulation(start_idle_times, final_assembly_parts, graph, part_data, makes, buys, deltas, env=simpy.Environment(), counter=1,):
    resources = {}
    states = {}
    evts = {str(name): Event(env) for name in list(graph.nodes())}
    completion_times = {}
    completion_queue = []
    delivery_times = {}
    last_part_before_make_begins = {}
    # Keep track of the final assembly parts and their completion status
    final_assembly_completion = {part: False for part in final_assembly_parts}
    log_lines = []
    end_event = simpy.Event(env)

    from enum import Enum

    class PART_STATE(str, Enum):
        WI = 'idle_wait'
        OFF = 'offset'
        ST = 'safety'
        GR = 'goods_received'
        PDT = 'planned_delivery'
        IHPT = 'inhouse_production'
        RLS = 'release'
        SMK = 'schedule_margin'
        PART_WAIT = 'wait_for_subparts'
        FIN = 'finish_idle'

    def trigger_done():
        if all(final_assembly_completion.values()):
            yield end_event.trigger()

    def set_state(name, PART_STATE, start_stop=True):
        log_lines.append(
            f"{env.now}:{name.upper()}:{'BEGIN' if start_stop else 'END'}:{str(PART_STATE)}")
        print(
            f"{env.now}:{name.upper()}:{'BEGIN' if start_stop else 'END'}:{str(PART_STATE)}")
        states[name] = PART_STATE

    def part_buy(env, name, start_idle_time, offset_time, safety_time, goods_receipt_time, planned_delivery_time, schedule_margin_time):
        set_state(name, PART_STATE.WI)
        yield env.timeout(start_idle_time)
        set_state(name, PART_STATE.WI, False)

        set_state(name, PART_STATE.SMK)
        yield env.timeout(schedule_margin_time)
        set_state(name, PART_STATE.SMK, False)

        set_state(name, PART_STATE.PDT)
        pdt_start = env.now
        delta_time = deltas[name](
            1)[0] if name in deltas else np.random.randint(-3, 30)
        planned_delivery_time = max(0, planned_delivery_time + delta_time)
        yield env.timeout(planned_delivery_time)
        pdt_end = env.now
        set_state(name, PART_STATE.PDT, False)
        delivery_times[name] = (pdt_end - pdt_start)

        set_state(name, PART_STATE.GR)
        yield env.timeout(goods_receipt_time)
        set_state(name, PART_STATE.GR, False)

        set_state(name, PART_STATE.ST)
        yield env.timeout(safety_time)
        set_state(name, PART_STATE.ST, False)

        set_state(name, PART_STATE.OFF)
        yield env.timeout(offset_time)
        set_state(name, PART_STATE.OFF, False)

        set_state(name, PART_STATE.RLS)
        yield evts[name].succeed()
        set_state(name, PART_STATE.RLS, False)

        set_state(name, PART_STATE.FIN)
        completion_times[name] = env.now
        completion_queue.append(('Buy', name))
        final_assembly_completion[name] = True

    def part_make(env, name, priors, start_idle_time, inhouse_production_time, schedule_margin_time):
        set_state(name, PART_STATE.WI)
        yield env.timeout(start_idle_time)
        set_state(name, PART_STATE.WI, False)

        set_state(name, PART_STATE.SMK)
        yield env.timeout(schedule_margin_time)
        set_state(name, PART_STATE.SMK, False)

        # Wait for all sub parts to be ready
        set_state(name, PART_STATE.PART_WAIT)
        yield AllOf(env, [evts[part] for part in priors])
        set_state(name, PART_STATE.PART_WAIT, False)

        last_part_before_make_begins[name] = (completion_queue[-1][1], "MAKE" if completion_queue[-1][1] in makes else "BUY", completion_times[completion_queue[-1][1]])

        set_state(name, PART_STATE.IHPT)
        yield env.timeout(inhouse_production_time)
        set_state(name, PART_STATE.IHPT, False)

        set_state(name, PART_STATE.RLS)
        yield evts[name].succeed()
        set_state(name, PART_STATE.RLS, False)

        set_state(name, PART_STATE.FIN)
        completion_times[name] = env.now
        completion_queue.append(('Make', name))
        final_assembly_completion[name] = True

    for i, node in enumerate(graph.nodes):
        name = str(node)
        # Get lt_offset and safety_time from the dataset if available
        part_info = part_data.get(name, {})
        start_idle_time = start_idle_times[i]

        def get_estimate(key):
            return max(1, part_info.get(key, random.uniform(1, 30)) + np.random.normal(part_info.get(key, random.uniform(1, 30)), 15))

        env.process(
            part_buy(
                env, name,
                start_idle_time=start_idle_time,
                # Use lt_offset from dataset if available
                offset_time=get_estimate('lt_offset'),
                # Use safety_time from dataset if available
                safety_time=get_estimate('safety_time'),
                goods_receipt_time=get_estimate('goods_receipt_time'),
                planned_delivery_time=get_estimate(
                    'planned_delivery_time'),
                schedule_margin_time=get_estimate('schedule_margin_time'),
            ) if name in buys else
            part_make(
                env, name, [str(n) for n in graph.predecessors(name)],
                start_idle_time=start_idle_time,
                inhouse_production_time=get_estimate('planned_delivery_time'),
                schedule_margin_time=get_estimate('schedule_margin_time'))
        )

    env.run(until=None)
    log_df = pd.read_csv(io.StringIO('\n'.join(log_lines)), sep=':', names=[
                         'Time', 'Part', 'Begin_End', 'State', ]).drop_duplicates()
    completion_df = pd.DataFrame(
        completion_times.items(), columns=['Part', 'Completion'])
    delivery_df = pd.DataFrame(
        delivery_times.items(), columns=['Part', 'Delivery'])
    critical_df = pd.DataFrame(last_part_before_make_begins.items(), columns=[
                               'Make', 'Critical Part'])
    log_df['Run'] = counter
    delivery_df['Run'] = counter
    critical_df['Run'] = counter
    critical_df['Predecessor'] = critical_df['Critical Part'].apply(lambda x: x[0])
    critical_df['Make_Buy'] = critical_df['Critical Part'].apply(lambda x: x[1])
    critical_df['Time'] = critical_df['Critical Part'].apply(lambda x: x[2])
    critical_df.drop(['Critical Part'], axis=1, inplace=True)
    completion_df['Run'] = counter
    return completion_df, delivery_df, critical_df, log_df, collect_last_nodes_per_level_reverse(graph, final_assembly_parts[0], completion_times).Part.tolist()
