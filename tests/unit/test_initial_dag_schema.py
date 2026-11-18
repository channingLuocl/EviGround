"""Initial joint generation enforces existing graph rules before runtime checks."""
from copy import deepcopy
from itertools import permutations
import unittest

from jsonschema import Draft202012Validator

from eviground.grounding import react
from eviground.grounding.dag_monitor import DagMonitor
from tests.helpers import parsed
from tests.unit import test_dag_delta
from tests.unit.test_dag_monitor import QUERY, action, node, plan


class InitialDagSchemaTests(unittest.TestCase):
    def setUp(self):
        self.monitor = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2},
                                  ['inspect', 'measure', 'finish'])
        self.original = test_dag_delta.compact_ids(plan())
        self.original['nodes'][-1]['depends_on'] = ['n0', 'n1']

    def compiled(self):
        helper = test_dag_delta.DagDeltaTests()
        helper.setUp()
        _, calls = helper.run_ground([
            action() | {'monitor': self.monitor.encode(self.original)},
            action('finish', None) | {'monitor': self.monitor.encode(self.original)}])
        return calls[0].kwargs['response_schema']['schema']

    def test_joint_entrypoint_rejects_initial_missing_dependency(self):
        schema = self.compiled()
        Draft202012Validator.check_schema(schema)
        value = action() | {'monitor': self.monitor.encode(self.original)}
        validator = Draft202012Validator(schema)
        validator.validate(value)
        for change in ('missing_dependency', 'absent_assignment', 'forward_edge',
                       'duplicate_id', 'absent_current', 'first_plan_reopen'):
            invalid = deepcopy(value)
            monitor = invalid['monitor']
            if change == 'missing_dependency':
                monitor['verification']['depends_on'] = ['n0']
            elif change == 'absent_assignment':
                monitor['requirement_nodes']['q0'] = ['n10']
            elif change == 'forward_edge':
                monitor['nodes'][0]['depends_on'] = ['n1']
            elif change == 'duplicate_id':
                monitor['nodes'][1]['id'] = 'n0'
            elif change == 'absent_current':
                monitor['current_node_id'] = 'n10'
            else:
                monitor['reopen_node_ids'] = ['n0']
            with self.subTest(change=change):
                self.assertFalse(validator.is_valid(invalid))

    def test_every_permitted_task_count_remains_expressible(self):
        schema = self.compiled()
        validator = Draft202012Validator(schema)
        for count in range(1, 12):
            tasks = [node(f'n{i}', 'binding', [f'n{i-1}'] if i else ())
                     for i in range(count)]
            value = {'nodes': tasks + [node('n11', 'verification', [n['id'] for n in tasks])],
                     'current_node_id': 'n0', 'reopen_node_ids': [],
                     'revision_reason': 'Resolve the description using the chosen task count'}
            with self.subTest(count=count):
                wire = self.monitor.encode(value)
                validator.validate(action() | {'monitor': wire})
                self.monitor.update(self.monitor.decode(wire), action(), [])

    def test_canonical_numbering_preserves_graph_and_rollback_reachability(self):
        # Every valid three-task DAG, including every label order, can be encoded.
        schema = self.compiled()
        validator = Draft202012Validator(schema)
        for order in permutations(('task_a', 'task_b', 'task_c')):
            for edges in range(8):
                possible = [(0, 1), (0, 2), (1, 2)]
                tasks = [node(name, 'binding') for name in order]
                for bit, (before, after) in enumerate(possible):
                    if edges & (1 << bit):
                        tasks[after]['depends_on'].append(order[before])
                used_as_prerequisite = {name for task in tasks for name in task['depends_on']}
                leaves = [name for name in order if name not in used_as_prerequisite]
                original = {'nodes': tasks + [node('verify', 'verification', leaves)],
                            'current_node_id': order[0], 'reopen_node_ids': [],
                            'revision_reason': 'Preserve the same tasks and dependency reachability'}
                rename = {name: f'n{i}' for i, name in enumerate(order)} | {'verify': 'n11'}
                normalized = deepcopy(original)
                for row in normalized['nodes']:
                    row['id'] = rename[row['id']]
                    row['depends_on'] = [rename[name] for name in row['depends_on']]
                normalized['nodes'][-1]['depends_on'] = [rename[name] for name in order]
                normalized['current_node_id'] = rename[original['current_node_id']]
                validator.validate(action() | {'monitor': self.monitor.encode(normalized)})
                for name in order:
                    before = self.monitor.descendants(original['nodes'], {name})
                    after = self.monitor.descendants(normalized['nodes'], {rename[name]})
                    self.assertEqual({rename[value] for value in before}, after)


if __name__ == '__main__':
    unittest.main()
