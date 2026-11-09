import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
class AppTests(unittest.TestCase):
 def test_pages_prediction_test_and_history(self):
  app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
  self.assertFalse(app.exception)
  for page in ['Dự đoán một mẫu','Sàng lọc từ CSV','Kiểm chứng test','So sánh mô hình','Đọc ảnh bảng đo','Lịch sử','Thông tin đề tài','Tổng quan']:
   app.sidebar.radio[0].set_value(page).run(timeout=60)
   self.assertFalse(app.exception,page)
  app.sidebar.radio[0].set_value('Dự đoán một mẫu').run()
  app.selectbox[0].set_value('Mẫu có hỏng').run()
  app.button[0].click().run(timeout=60)
  self.assertFalse(app.exception)
  self.assertEqual(int(app.session_state['single_result'].iloc[0].prediction),1)
  self.assertEqual(len(app.session_state['history']),1)
  app.sidebar.radio[0].set_value('Kiểm chứng test').run()
  app.button[0].click().run(timeout=60)
  self.assertFalse(app.exception)
  self.assertEqual(len(app.session_state['test_result']),1500)
  app.selectbox[0].set_value('Sai · Bỏ sót').run()
  self.assertFalse(app.exception)
  app.sidebar.radio[0].set_value('Lịch sử').run()
  app.button[0].click().run()
  self.assertEqual(len(app.session_state['history']),0)

 def test_outside_pair(self):
  app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
  app.sidebar.radio[0].set_value('Dự đoán một mẫu').run()
  app.number_input[0].set_value(444.)
  app.number_input[1].set_value(477.)
  app.button[0].click().run()
  self.assertFalse(app.exception)
  self.assertIn('không đủ cơ sở kết luận',app.warning[0].value)
  self.assertFalse(app.success)
  self.assertFalse(app.metric)
  self.assertTrue(app.session_state['single_result'].prediction.isna().all())
  self.assertTrue(app.session_state['single_result'].probability.isna().all())

 def test_mixed_csv_has_explicit_outcome(self):
  import pandas as pd
  data=pd.read_csv(ROOT/'data/mau_nhap.csv').iloc[:3].copy()
  data.iloc[1,data.columns.get_loc('Air temperature [K]')]=444
  data.iloc[2,data.columns.get_loc('Type')]='Z'
  app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
  app.sidebar.radio[0].set_value('Sàng lọc từ CSV').run()
  app.main.radio[0].set_value('Tải CSV của bạn').run()
  app.file_uploader[0].upload('thu.csv',data.to_csv(index=False).encode(),'text/csv').run()
  app.button[0].click().run()
  self.assertFalse(app.exception)
  result=app.session_state['batch_result']
  self.assertTrue(result.prediction.notna().tolist()==[True,False,False])
  self.assertIn('không đủ cơ sở kết luận',result.iloc[1]['Kết luận'])
  self.assertEqual(result.iloc[2]['Kết luận'],'Dữ liệu không hợp lệ')
  self.assertTrue(result.loc[1:, 'Trường hợp'].isna().all())

 def test_priority_queue_and_cost_simulation(self):
  app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
  app.button[0].click().run()
  self.assertFalse(app.exception)
  queue=app.dataframe[0].value
  self.assertGreater(len(queue), 0)
  self.assertTrue(queue['Điểm lớp hỏng'].is_monotonic_decreasing)
  self.assertTrue(queue.prediction.eq(1).all())
  app.sidebar.radio[0].set_value('So sánh mô hình').run()
  original=app.dataframe[1].value.copy()
  app.number_input[0].set_value(10).run()
  self.assertFalse(app.exception)
  self.assertNotEqual(app.dataframe[1].value.loc[0, 'Chi phí quy ước'],
                      original.loc[0, 'Chi phí quy ước'])

 def test_image_requires_review_before_prediction(self):
  import shutil
  if not shutil.which('tesseract'):
   self.skipTest('Tesseract is not installed.')
  app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
  app.sidebar.radio[0].set_value('Đọc ảnh bảng đo').run()
  app.button[0].click().run()
  self.assertFalse(app.exception)
  self.assertTrue(all(app.session_state['ocr_fields'].values()))
  app.button[1].click().run()
  self.assertTrue(app.warning)
  app.checkbox[0].check()
  app.button[1].click().run()
  self.assertFalse(app.exception)
  self.assertEqual(int(app.session_state['ocr_result'].iloc[0].prediction), 1)
