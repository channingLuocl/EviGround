"""Guard the package migration against missing modules and inference changes."""
import ast
import copy
import json
from pathlib import Path
import unittest

from eviground import paths


REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / 'outputs/r18_qwen_original_resume_20261022/runtime_original/local'


class NormalizeImportsAndPaths(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ('load_dotenv', 'load_environment')):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == 'ROOT'
               for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace('local/validation', 'data/validation')
            node.value = node.value.replace('EVIGROUND_', 'SEEGROUND_')
            node.value = node.value.replace('eviground', 'seeground').replace('EviGround', 'SeeGround')
        return node


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = json.loads((paths.PACKAGE_ROOT / '_module_layout.json').read_text())

    def test_all_frozen_modules_have_unique_destinations(self):
        self.assertEqual(set(self.layout), {p.name for p in FROZEN.glob('*.py')})
        self.assertEqual(len(self.layout), len(set(self.layout.values())))
        for original, destination in self.layout.items():
            with self.subTest(original=original):
                self.assertEqual(paths.source_file(original), paths.PACKAGE_ROOT / destination)
                self.assertTrue(paths.source_file(original).is_file())

    def test_internal_import_modules_exist(self):
        for source in paths.source_files():
            tree = ast.parse(source.read_text(), filename=str(source))
            for node in ast.walk(tree):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                           [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    if module and (module == 'eviground' or module.startswith('eviground.')):
                        target = paths.PACKAGE_ROOT.parent / module.replace('.', '/')
                        with self.subTest(source=source.name, line=node.lineno, module=module):
                            self.assertTrue(target.with_suffix('.py').is_file() or
                                            (target / '__init__.py').is_file())

    def test_inference_modules_match_frozen_baseline(self):
        # Only these two functions change resource/fingerprint paths; no decision logic.
        path_metadata = {'v5_runtime.py': {'input_identity'}, 'shape_scale.py': {'input_config'}}
        core = {'api', 'categories', 'grounding', 'query', 'geometry', 'rendering'}
        for original, destination in self.layout.items():
            if destination.split('/')[0] not in core:
                continue
            before = ast.parse((FROZEN / original).read_text())
            after = ast.parse((paths.PACKAGE_ROOT / destination).read_text())
            for tree in (before, after):
                tree.body = [node for node in tree.body
                             if not (isinstance(node, ast.FunctionDef) and
                                     node.name in path_metadata.get(original, set()))]
            with self.subTest(module=original):
                self.assertEqual(ast.dump(NormalizeImportsAndPaths().visit(copy.deepcopy(before))),
                                 ast.dump(NormalizeImportsAndPaths().visit(copy.deepcopy(after))))

    def test_resources_and_frozen_validation_alias(self):
        selection = paths.VALIDATION_ROOT / 'nr3d_val_full.json'
        self.assertEqual(len(json.loads(selection.read_text())['rows']), 7805)
        alias = REPO / 'reproductions/SeeGround/local/validation/nr3d_val_full.json'
        self.assertEqual(alias.resolve(), selection.resolve())
        self.assertTrue(paths.source_file('parts_v83.json').is_file())
        self.assertTrue(paths.source_file('category_v10/run.py').is_file())
        self.assertTrue((paths.ROOT / 'third_party/models/pcd_classifier.py').is_file())

    def test_source_fingerprints_use_unique_relative_paths(self):
        sources = paths.source_files()
        identities = [p.relative_to(paths.PACKAGE_ROOT).as_posix() for p in sources]
        self.assertEqual(len(identities), len(set(identities)))
        from eviground.classifiers.category_v10.run import code_hashes
        self.assertEqual(set(code_hashes()), set(identities))


if __name__ == '__main__':
    unittest.main()
