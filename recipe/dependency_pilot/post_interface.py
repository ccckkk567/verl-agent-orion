"""Public ScienceWorld interface and observer-only evidence; no reward changes."""
import json
import re
from .common import digest

PROFILES = ('p0_original', 'reference_interface_v1', 'reference_interface_endpoint_v1')
COMMAND_GUIDE = """Command meanings (OBJ is a reference from the objects below):
open/close: change a container or door; activate/deactivate: switch a device.
look around: inspect the room; look at OBJ: inspect an object; look in OBJ: inspect contents.
go OBJ (also go to OBJ): travel to a location; a closed door must be opened first.
pick up/put down: take/drop; move OBJ to OBJ: place in a container.
connect OBJ to OBJ: join electrical parts; disconnect OBJ: remove its connections.
focus on OBJ: select a task object as requested by the task, not a navigation command.
read/use/mix/pour/dunk/eat/flush operate on their named objects; inventory lists carried items.
wait1 advances one tick; wait advances ten. task repeats the task description.
Templates and object references are syntax aids, not guarantees that an action will work."""
ENDPOINT_NOTE = ('Electrical connections can require component endpoints rather than whole components. '
                 'Use exact endpoint references from the public object list or details obtained by looking at an object.')

def public_context(env, profile):
    templates = list(env.get_possible_actions())
    objects = sorted(set(env.get_possible_objects()))
    # Fixed lexical cap, independent of goal, reward, and hidden state.
    rendered = objects[:80]
    block = (COMMAND_GUIDE + '\nObject references (lexical order, at most 80):\n'
             + '\n'.join(rendered))
    if profile == PROFILES[2]:
        block += '\nGeneral endpoint interface note:\n' + ENDPOINT_NOTE
    return block, {'action_templates_source': 'ScienceWorldEnv.get_possible_actions',
        'object_refs_source': 'ScienceWorldEnv.get_possible_objects',
        'object_refs_count': len(objects), 'object_refs_rendered_count': len(rendered),
        'object_refs_omitted_count': len(objects)-len(rendered), 'object_refs_rendered': rendered,
        'available_context_hash': digest(json.dumps([templates, rendered, block])),
        'endpoint_guidance_version': 'generic_endpoint_v1' if profile == PROFILES[2] else None,
        'simplifications_expanded': [], 'actor_interface_block': block}

def electrical_snapshot(env):
    """Read-only object tree, never actor-visible; normalize undirected endpoint edges."""
    try:
        tree = env.getObjectTree()
        if isinstance(tree, str): tree = json.loads(tree)
        nodes, edges = {}, set()
        def walk(node, parent=''):
            uid = str(node['uuid'])
            nodes[uid] = {'name': node['name'], 'parent': parent}
            prop = node.get('propElectricalConnection')
            if prop is not None:
                for other in prop.get('connectedTo', []):
                    edges.add(tuple(sorted((uid, str(other)))))
            for child in node.get('contents', {}).values():
                walk(child, node['name'])
        walk(tree)
        return {'edges': [list(e) for e in sorted(edges)],
                'endpoints': {u: nodes.get(u) for u in sorted({u for e in edges for u in e})},
                'moves_after_read': env.get_num_moves(), 'error': None}
    except Exception as exc:
        return {'edges': None, 'endpoints': None, 'error': f'{type(exc).__name__}: {exc}'}

def parser_metadata(raw, action, env):
    # Diagnostic only: preserve original parser and actual command, including fallback.
    flags = re.S | (re.I if env == 'scienceworld' else 0)
    blocks = re.findall(r'<action>(.*?)</action>', raw, flags)
    clean = bool(blocks) and bool(action.strip()) and '\n' not in action and not re.search(r'</?(?:action|think)>',action,re.I)
    unique = len(blocks) == 1 and len(re.findall(r'<action>',raw,re.I)) == 1 and len(re.findall(r'</action>',raw,re.I)) == 1
    return {'parser_usable': bool(clean), 'response_action_structure_unambiguous':bool(clean and unique), 'parser_fallback_used': not bool(blocks),
            'action_after_parser': action, 'parser_profile': 'p0_original_unchanged'}

def execution_outcome(action, feedback, before=None, after=None):
    lower = feedback.lower().strip()
    if lower == 'no known action matches that input.': return 'unmatched_command'
    if any(s in lower for s in ['which one do you mean', 'need to specify', 'specify which', 'not clear which', 'connections must specify specific points of contact']):
        return 'object_or_endpoint_resolution_failed'
    if any(s in lower for s in ['door is not open', 'must be disconnected', 'already connected']):
        return 'precondition_blocked'
    if before and after and before.get('edges') is not None and after.get('edges') is not None:
        if before['edges'] != after['edges']: return 'accepted_state_change'
    if (' is now connected to ' in lower or lower.startswith('you move to ') or
        lower in ['the door is now open.', 'the door is now closed.']):
        return 'accepted_state_change'
    if action.lower().startswith(('look ', 'inventory', 'task')) and feedback.strip() and not any(
            s in lower for s in ['no known', 'could not', 'cannot', 'unable', 'not found']):
        return 'accepted_read_or_observation'
    return 'unknown'
