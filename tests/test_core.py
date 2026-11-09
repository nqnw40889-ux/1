import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from core import NUM,FEATURES,validate,predict_batch

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(42);n=120
        cls.x=pd.DataFrame({'Type':rng.choice(['L','M','H'],n),NUM[0]:rng.uniform(295,305,n),NUM[1]:rng.uniform(305,315,n),NUM[2]:rng.integers(1200,2500,n),NUM[3]:rng.uniform(10,70,n),NUM[4]:rng.integers(0,250,n)})[FEATURES]
        y=(cls.x[NUM[3]]>40).astype(int)
        prep=ColumnTransformer([('num','passthrough',NUM),('type',OneHotEncoder(handle_unknown='error'),['Type'])])
        pipe=Pipeline([('preprocessing',prep),('model',LogisticRegression(max_iter=2000))]).fit(cls.x,y)
        cls.bundle={'pipeline':pipe,'threshold':.4,'feature_columns':FEATURES}
        cls.meta={'train_ranges':{c:{'min':float(cls.x[c].min()),'max':float(cls.x[c].max())} for c in NUM}}
    def test_prediction_roundtrip_and_original_index(self):
        a=self.x.iloc[:3].copy();a.index=[4,8,15]
        result=predict_batch(self.bundle,a,self.meta)
        expected=self.bundle['pipeline'].predict_proba(a)[:,1]
        np.testing.assert_allclose(result.probability,expected)
        self.assertEqual(result.prediction.tolist(),(expected>=.4).astype(int).tolist())
    def test_missing_column(self):
        with self.assertRaises(ValueError):validate(self.x.drop(columns='Type'),self.meta)
    def test_invalid_values(self):
        x=self.x.iloc[:4].copy().reset_index(drop=True)
        x[NUM[2]]=x[NUM[2]].astype(float)
        x.loc[0,'Type']='Z';x.loc[1,NUM[0]]=np.inf;x.loc[2,NUM[4]]=-1;x.loc[3,NUM[2]]=1400.5
        result=predict_batch(self.bundle,x,self.meta)
        self.assertTrue(result.prediction.isna().all());self.assertTrue(result.input_error.ne('').all())
    def test_outside_warns_and_predicts_by_default(self):
        x=self.x.iloc[:1].copy();x[NUM[0]]=1000
        r=predict_batch(self.bundle,x,self.meta,False)
        self.assertTrue(r.prediction.isna().all());self.assertTrue(r.range_warning.ne('').all())
        self.assertTrue(predict_batch(self.bundle,x,self.meta).prediction.notna().all())
    def test_mixed_batch_not_all_or_nothing(self):
        x=self.x.iloc[:2].copy().reset_index(drop=True);x.loc[1,NUM[1]]=np.nan
        r=predict_batch(self.bundle,x,self.meta)
        self.assertTrue(pd.notna(r.loc[0,'prediction']));self.assertTrue(pd.isna(r.loc[1,'prediction']))
    def test_reload(self):
        import tempfile,joblib
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.joblib';joblib.dump(self.bundle,path)
            np.testing.assert_allclose(predict_batch(self.bundle,self.x.iloc[:2],self.meta).probability,predict_batch(joblib.load(path),self.x.iloc[:2],self.meta).probability)
if __name__=='__main__':unittest.main()
