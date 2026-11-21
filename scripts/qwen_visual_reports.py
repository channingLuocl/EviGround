"""Three-level visual evidence: only vision workers receive original image bytes."""
from copy import deepcopy
import re

from jsonschema import validate

from qwen_evidence_selector import EvidenceSelectionError, registry
from qwen_step2_context import packed

COORDINATOR = 'visual_coordinator'
WORKER = 'visual_worker'
TEXT_RULES = """
You receive text only. A visual finding covers only its listed image, evidence
and object IDs. Consider conflicting findings. Occlusion, missing context and
ambiguous identity remain unknown; do not use them to exclude. Only delivered
findings and raw measurements support new conclusions. Use action.evidence_request
for a specific question about an existing image if the schema permits it.
Use action.read_evidence_page for one indexed archived report or measurement.
The archive index is for lookup; only delivered source data counts as evidence.
Use inspect for a new camera. In a tentative finish, state remaining gaps.
"""
PLANNER_RULES = """
You coordinate visual evidence by reading text. Split the request into one
concrete visual question per supplied image. Return exactly one task for each
supplied image ID. Name the object IDs to compare when known. Do not answer the
questions, infer a target or update the grounding plan. For isolated views,
do not ask the reader to infer room adjacency from tile placement.
"""
WORKER_RULES = """
Read the ORIGINAL image for the assigned question. Give one short literal
sentence per relevant finding, focused on the question. Use the shown labels only.
List only object IDs with labels in this image in an observed finding. For
unknown findings, include requested IDs that could not be checked.
If a relation, identity or visibility cannot be determined, report unknown and
explain what is missing. Occlusion or absence from a view cannot prove an
object is absent or that a relation is false. Separate isolated tiles do not
establish room position. Report what this view supports, without a target
decision, candidate exclusion or Monitor update.
Return one JSON object with the assigned image_id and the findings array.
"""
REVIEW_RULES = """
You coordinate the returned TEXT observations. Group findings about the same
objects and relation or attribute, covering every supplied finding. List each
finding ID at most once per group. A finding may support more than one group.
Use only as many groups as there are supplied findings.
Mark agreement as consistent only when every finding in the group is observed.
Mark disagreement as conflicting and gaps as insufficient. Keep
conflicts and unknowns visible. Do not generate extra observations, select a
target or change the grounding plan.
"""


def obj(fields):
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def array(items):
    return {'type': 'array', 'items': items}


def one_of(values):
    return {'enum': list(values)}


TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 480}
IDS = array({'type': 'integer'})
SOURCE_IDS = re.compile(r'\b(?:img_[0-9]+|visual_[0-9]+(?:_f[0-9]+)?)\b')


def request_schema(image_ids):
    return obj({'image_ids': {**array(one_of(image_ids)), 'minItems': 1, 'uniqueItems': True},
                'question': TEXT})


class VisualReports:
    """Sample-local registry, reports, and explicit image rechecks."""

    def __init__(self, call, save, *, coordinated=True):
        self.call = call
        self.save = save
        self.coordinated = coordinated
        self.views = {}
        self.reports = {}
        self.completed = set()
        self.rechecks = set()
        self.recheck_reports = set()
        self.current_recheck_reports = set()
        self.current_readback = None
        self.count = 0

    def sync(self, state, messages):
        full, units = registry(state, messages)
        current = {}
        for unit in units:
            for index, image, sha in zip(unit['source_view_indices'], unit['images'], unit['image_sha256'], strict=True):
                image_id = f'img_{index + 1:04d}'
                current[image_id] = {'image_id': image_id, 'evidence_id': unit['id'], 'sha256': sha,
                                     'view': full['views'][index], 'image': image}
                if image_id in self.views and current[image_id] != self.views[image_id]:
                    raise EvidenceSelectionError(f'Original image identity changed for {image_id}')
        if self.views.keys() - current.keys():
            raise EvidenceSelectionError('Original image registry shrank')
        self.views = current
        return units

    def catalog(self):
        return [{k: row[k] for k in ('image_id', 'evidence_id', 'sha256')} for row in self.views.values()]

    def collect_new(self, state, messages):
        units = self.sync(state, messages)
        for unit in units:
            group = [image_id for image_id, row in self.views.items() if row['evidence_id'] == unit['id']]
            if group and unit['id'] not in self.completed:
                self.collect(state, group, 'What does this image show about the description and the active candidates?')
                self.completed.add(unit['id'])
        return units

    def recheck(self, state, request):
        validate(request, request_schema(self.views))
        question = request['question'].strip()
        key = (tuple(sorted(request['image_ids'])), ' '.join(question.split()).casefold())
        if not question or key in self.rechecks:
            raise ValueError('Repeated or blank image recheck. Choose a different question or use native inspect.')
        self.rechecks.add(key)
        previous = set(self.reports)
        self.collect(state, request['image_ids'], question)
        new_reports = self.reports.keys() - previous
        self.recheck_reports.update(new_reports)
        self.current_recheck_reports.update(new_reports)

    def collect(self, state, image_ids, question):
        # A native render produces at most two images; long explicit reviews are
        # dispatched as separate pairs to keep intermediary requests small.
        for offset in range(0, len(image_ids), 2):
            self.collect_batch(state, image_ids[offset:offset + 2], question)

    def collect_batch(self, state, image_ids, question):
        self.count += 1
        run_id = f'visual_{self.count:04d}'
        known = {row['id'] for row in state['objects']}
        active = state.get('active_target_ids', state.get('allowed_target_ids', []))
        context = {'description': state['description'], 'question': question, 'active_target_ids': active,
                   'ready_node_ids': state.get('ready_node_ids', []),
                   'reference_candidates': state.get('reference_category_bindings', []),
                   'images': [{key: self.views[image_id][key] for key in ('image_id', 'evidence_id', 'view')}
                              for image_id in image_ids]}
        task_schema = obj({'image_id': one_of(image_ids), 'question': TEXT, 'object_ids': IDS})
        if self.coordinated:
            plan_schema = obj({'tasks': {**array(task_schema), 'minItems': len(image_ids), 'maxItems': len(image_ids)}})
            plan = self.ask(run_id + '-plan', COORDINATOR, PLANNER_RULES, context, plan_schema, 1024)
        else:
            plan = {'tasks': [{'image_id': image_id, 'question': question, 'object_ids': active} for image_id in image_ids]}
        if {task['image_id'] for task in plan['tasks']} != set(image_ids):
            raise EvidenceSelectionError('Coordinator did not dispatch exactly one task per requested image')
        self.save(run_id, 'plan.json', {'request': context, 'plan': plan})

        findings = []
        for task in plan['tasks']:
            # A planner may list the same ID as both a target and a reference.
            task = {**task, 'object_ids': list(dict.fromkeys(task['object_ids']))}
            self.check_objects(task, known)
            image_id = task['image_id']
            row = self.views[image_id]
            marked = set(row['view'].get('marked_ids', []))
            finding = obj({'object_ids': IDS, 'status': one_of(['observed', 'unknown']), 'observation': TEXT})
            schema = obj({'image_id': {'const': image_id}, 'findings': {**array(finding), 'minItems': 1, 'maxItems': 6}})
            result = self.ask(run_id + '-' + image_id, WORKER, WORKER_RULES,
                              {'description': state['description'], 'task': task,
                               'evidence_id': row['evidence_id'], 'view': row['view'],
                               'known_object_ids': sorted(known)}, schema, 1024, row['image'])
            for raw in result['findings']:
                self.check_objects(raw, known)
                if raw['status'] == 'observed' and set(raw['object_ids']) - marked:
                    raise EvidenceSelectionError('Visual worker claimed an observed object ID without a label in its image')
                findings.append({**raw, 'finding_id': f'{run_id}_f{len(findings) + 1:02d}',
                                 'image_id': image_id, 'image_sha256': row['sha256'],
                                 'evidence_id': row['evidence_id'], 'question': task['question']})
        self.save(run_id, 'findings.json', findings)
        if not self.coordinated:
            report = {'report_id': run_id, 'question': question, 'findings': findings, 'groups': []}
            self.save(run_id, 'report.json', report)
            self.reports[run_id] = report
            return
        by_id = {row['finding_id']: row for row in findings}
        group = obj({'assessment': one_of(['consistent', 'conflicting', 'insufficient']),
                     'finding_ids': {**array(one_of(by_id)), 'minItems': 1, 'maxItems': len(by_id)}})
        schema = obj({'groups': {**array(group), 'minItems': 1, 'maxItems': len(by_id)}})
        review = self.ask(run_id + '-review', COORDINATOR, REVIEW_RULES,
                          {'question': question, 'findings': [{key: value for key, value in row.items()
                             if key not in ('image_sha256', 'evidence_id', 'question')} for row in findings]}, schema, 3072)
        grouped = {key for group in review['groups'] for key in group['finding_ids']}
        if missing := sorted(by_id.keys() - grouped):
            review['groups'].append({'assessment': 'insufficient', 'finding_ids': missing})
        for group in review['groups']:
            group['finding_ids'] = list(dict.fromkeys(group['finding_ids']))
            rows = [by_id[key] for key in group['finding_ids']]
            if group['assessment'] == 'consistent' and any(row['status'] == 'unknown' for row in rows):
                group['assessment'] = 'insufficient'
        report = {'report_id': run_id, 'question': question, 'findings': findings, 'groups': review['groups']}
        self.save(run_id, 'report.json', report)
        self.reports[run_id] = report

    @staticmethod
    def check_objects(row, known):
        ids = row['object_ids']
        if set(ids) - known or len(ids) != len(set(ids)):
            raise EvidenceSelectionError('Visual report or task names an unknown or duplicate object ID')

    def ask(self, tag, role, rules, context, schema, reserve, image=None):
        content = packed(context) if image is None else [{'type': 'text', 'text': packed(context)}, deepcopy(image)]
        answer = self.call(tag, role, [{'role': 'system', 'content': rules},
                                       {'role': 'user', 'content': content}], schema, reserve)
        return answer

    def selected_reports(self, evidence_ids, omitted_reports):
        # Only ship complete reviews. Report-level provenance never points into
        # another report or an image that did not actually reach the reader.
        return [report for report in self.reports.values()
                if report['report_id'] not in omitted_reports and
                {f['evidence_id'] for f in report['findings']} <= evidence_ids]

    def reports_for_citations(self, text):
        cited = set(SOURCE_IDS.findall(text))
        return {report_id for report_id, report in self.reports.items()
                if report_id in cited or any(row['finding_id'] in cited or row['image_id'] in cited
                                             for row in report['findings'])}

    @staticmethod
    def check_image_citations(answer, shown_reports):
        visible = {value for report in shown_reports for finding in report['findings']
                   for value in (report['report_id'], finding['finding_id'], finding['image_id'])}
        def check(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ('reason', 'observation', 'conclusion', 'comparison') and isinstance(child, str):
                        missing = set(SOURCE_IDS.findall(child)) - visible
                        if missing:
                            raise ValueError(f'Claim cites visual sources absent from delivered reports: {sorted(missing)}')
                    elif isinstance(child, (dict, list)):
                        check(child)
            elif isinstance(value, list):
                for child in value: check(child)
        check(answer)
