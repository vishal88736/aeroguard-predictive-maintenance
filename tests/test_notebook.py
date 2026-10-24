import ast
import json
from pathlib import Path
import unittest

from fleetpm.config import ROOT


class NotebookTests(unittest.TestCase):
    def setUp(self):self.book=json.loads((ROOT/'notebooks/C_MAPSS_All_Subsets_CUDA.ipynb').read_text())

    def test_unexecuted_valid_code_cells(self):
        self.assertEqual(self.book['nbformat'],4)
        codes=[c for c in self.book['cells'] if c['cell_type']=='code'];self.assertGreater(len(codes),10)
        for i,cell in enumerate(codes):
            self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
            ast.parse(''.join(cell['source']),filename=f'notebook-cell-{i}')

    def test_all_subsets_cuda_and_manual_install_contract(self):
        code='\n'.join(''.join(c['source']) for c in self.book['cells'] if c['cell_type']=='code')
        self.assertIn('SELECTED_SUBSETS = list(SUBSETS)',code);self.assertIn('os.environ.get("FLEET_TRAIN_DEVICE", "cuda")',code)
        self.assertIn('train_subset',code);self.assertIn('evaluate_run',code);self.assertIn('activate_run',code)
        self.assertNotIn('pip install',code);self.assertNotIn('subprocess',code)

    def test_training_module_does_not_reference_official_test_inputs(self):
        code=(ROOT/'fleetpm/train.py').read_text()
        self.assertNotIn('f"test_',code);self.assertNotIn('f"RUL_',code)
        self.assertNotIn('parse_truth',code)
