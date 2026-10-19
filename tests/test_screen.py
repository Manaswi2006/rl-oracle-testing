import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import screen
class TestScreen(unittest.TestCase):
 def test_success(self):
  for c in range(screen.NCTX):
   for t in range(screen.NSTAGE):self.assertIn(screen.correct_action(c,t),range(screen.NACT))
 def test_psvm(self):
  vals=screen.simulate_psvm(0);self.assertEqual(len(vals),12)
  for r in vals:self.assertTrue(0<=r['value']<=1)
 def test_facr(self):
  vals=screen.simulate_facr(0);self.assertEqual(len(vals),5)
  self.assertAlmostEqual(vals[1]['value'],vals[2]['value'])
 def test_falsification(self):
  vals=screen.simulate_falsification(0)
  for i in range(4):self.assertAlmostEqual(vals[4+i]['value'],vals[12+i]['value'])
if __name__=='__main__':unittest.main()
