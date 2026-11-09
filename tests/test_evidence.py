import unittest,sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import FEATURES,NUM,probability,predict_batch
from evaluation import measure
ROOT=Path(__file__).resolve().parents[1]
class EvidenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.b=joblib.load(ROOT/'mo_hinh_demo.joblib');cls.m=cls.b['metadata']
  cls.t=pd.read_csv(ROOT/'data/test_goc.csv')
 def test_split_identity_and_no_overlap(self):
  ids=[]
  for split,n in [('train',7000),('validation',1500),('test',1500)]:
   x=pd.read_csv(ROOT/f'data/{split}_goc.csv');self.assertEqual(len(x),n);ids.append(set(x.source_row))
   if split=='train':
    for c in NUM:self.assertEqual(x[c].min(),self.m['train_ranges'][c]['min']);self.assertEqual(x[c].max(),self.m['train_ranges'][c]['max'])
  self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]);self.assertEqual(len(set.union(*ids)),10000)
 def test_web_matches_training_and_labels_are_not_features(self):
  r=predict_batch(self.b,self.t,self.m)
  ref=pd.read_csv(ROOT/'reports/predictions_test.csv').set_index('source_row').loc[self.t.source_row]
  np.testing.assert_allclose(r.probability,ref.probability,atol=1e-12)
  changed=self.t.copy();changed['Machine failure']=1-changed['Machine failure']
  np.testing.assert_allclose(probability(self.b,changed),r.probability)
  self.assertEqual(list(self.b['feature_columns']),FEATURES)
  m=measure(self.t['Machine failure'],r.probability,self.b['threshold'])
  expected=pd.read_csv(ROOT/'reports/metrics.csv').set_index('Split').loc['Test']
  for k,v in m.items():self.assertAlmostEqual(v,expected[k],places=10)
 def test_treeshap_additivity_on_all_test(self):
  p=self.b.get('explanation_pipeline',self.b['pipeline']);z=p.steps[0][1].transform(self.t[FEATURES]);model=p.steps[-1][1]
  s=model.booster_.predict(z,pred_contrib=True);raw=model.booster_.predict(z,raw_score=True)
  np.testing.assert_allclose(s.sum(axis=1),raw,atol=1e-10)
  np.testing.assert_allclose(1/(1+np.exp(-raw)),p.predict_proba(self.t[FEATURES])[:,1],atol=1e-10)
