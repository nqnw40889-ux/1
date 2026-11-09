"""Nhóm 16 · AI4I research demo. Inference only; no training on the web."""
from pathlib import Path
from datetime import datetime, timezone
import io, csv, hashlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from core import FEATURES, NUM, LABELS, predict_batch, validate
from evaluation import measure
from ocr import read_sensor_image
from policy import cost_of, select_thresholds, threshold_curve

ROOT = Path(__file__).parent
st.set_page_config(page_title='Nhóm 16 | Machine Health', page_icon='⚙️', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'Be Vietnam Pro',sans-serif}
.stApp{background:#f3f6fa;color:#183449}.block-container{max-width:1250px;padding-top:2rem}
h1,h2,h3{letter-spacing:-.035em;color:#12374a} [data-testid="stSidebar"]{background:#102c3b}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h3{color:#e4f0f3!important}
.hero{background:linear-gradient(110deg,#102f40,#155568);border-radius:22px;padding:32px 36px;color:white;margin-bottom:24px;position:relative;overflow:hidden}
.hero:after{content:'⚙';position:absolute;right:28px;top:-24px;font-size:200px;color:#ffffff0b;line-height:1.2}
.hero h1{color:white;font-size:32px;margin:10px 0}.hero p{color:#c7dce5;max-width:750px;margin-bottom:0}.eyebrow{font-size:11px;font-weight:700;letter-spacing:2px;color:#76e0cb}
[data-testid="stMetric"]{background:white;border:1px solid #dfe8ef;border-radius:15px;padding:18px}
[data-testid="stMetricValue"]{font-size:27px} [data-testid="stForm"]{background:white;border:1px solid #dce6ee;border-radius:18px;padding:24px}
.stButton>button,.stDownloadButton>button{border-radius:10px}div[data-testid="stExpander"]{background:white;border-radius:12px}
.footer{font-size:12px;color:#647c8b;border-top:1px solid #dce5ed;padding-top:15px;margin-top:36px}
@media(max-width:700px){.hero{padding:22px}.hero h1{font-size:25px}.block-container{padding:1rem}}
</style>''', unsafe_allow_html=True)
PAGES=['Sàng lọc từ CSV','Kiểm chứng test','So sánh mô hình','Tổng quan','Dự đoán một mẫu','Đọc ảnh bảng đo','Lịch sử','Thông tin đề tài']
st.sidebar.title('⚙️ MACHINE HEALTH')
st.sidebar.caption('NHÓM 16  /  HỌC MÁY')
page=st.sidebar.radio('Điều hướng',PAGES)
st.sidebar.divider()
st.sidebar.markdown('### AI4I 2020')
st.sidebar.caption('Dự đoán hỏng hóc máy phục vụ bảo trì dự đoán trên dữ liệu cảm biến mất cân bằng.')
st.sidebar.caption('Demo nghiên cứu • Dữ liệu mô phỏng')

@st.cache_resource
def load_model(stamp):
    return joblib.load(ROOT/'mo_hinh_demo.joblib')
try:
    bundle=load_model((ROOT/'mo_hinh_demo.joblib').stat().st_mtime_ns)
    meta=bundle['metadata']
except Exception as e:
    st.error('Không tải được mô hình. Đặt mo_hinh_demo.joblib cạnh app.py và cài đúng requirements.txt.');st.code(str(e));st.stop()
threshold = float(bundle['threshold'])
miss_cost = int(meta['miss_cost'])
false_alarm_cost = int(meta['false_alarm_cost'])
POLICIES = {'Cost': 'Tối thiểu chi phí', 'F1': 'Tối đa F1', '0.5': 'Ngưỡng 0,5',
            'All normal': 'Luôn đoán không hỏng'}
OUTCOMES = ['Đúng · Không hỏng', 'Đúng · Có hỏng', 'Sai · Bỏ sót', 'Sai · Cảnh báo nhầm']
METRIC_LABELS = {
    'Model': 'Cấu hình', 'Policy': 'Cách chọn ngưỡng', 'Threshold': 'Ngưỡng',
    'Fit normal': 'Mẫu bình thường khi fit', 'Fit failure': 'Mẫu hỏng khi fit',
    'PR_AUC_trapezoid': 'PR-AUC', 'TP': 'Phát hiện đúng', 'FN': 'Bỏ sót',
    'FP': 'Báo nhầm', 'Cost': 'Chi phí quy ước',
}

st.session_state.setdefault('history',[])

def download(df,name,label='Tải kết quả CSV',key=None):
    st.download_button(label,df.to_csv(index=False).encode('utf-8-sig'),name,'text/csv',key=key)

def attach_labels(result):
    labels = pd.to_numeric(result['Machine failure'], errors='coerce')
    result['actual'] = labels.where(labels.isin([0, 1])).astype('Int64')
    correct = result.prediction.eq(result.actual)
    result['correct'] = correct
    result['Trường hợp'] = pd.Series(pd.NA, index=result.index, dtype='string')
    for actual, predicted, outcome in [(0, 0, OUTCOMES[0]), (1, 1, OUTCOMES[1]),
                                       (1, 0, OUTCOMES[2]), (0, 1, OUTCOMES[3])]:
        mask = result.actual.eq(actual) & result.prediction.eq(predicted)
        result.loc[mask.fillna(False), 'Trường hợp'] = outcome
    return result

def display_batch(result):
    first = [col for col in ['Dòng CSV', 'source_row', 'Kết luận', 'probability',
                             'Trường hợp', 'input_error', 'range_warning'] if col in result]
    columns = first + FEATURES + [col for col in ['Machine failure', 'prediction', 'threshold']
                                 if col in result]
    return result[columns].rename(columns={'probability': 'Điểm lớp hỏng',
                                           'input_error': 'Lỗi đầu vào',
                                           'range_warning': 'Cảnh báo phạm vi'})

def show_policies(frame):
    columns = ['Policy', 'Threshold', 'TP', 'FN', 'FP', 'Cost', 'F1']
    st.dataframe(frame[columns].rename(columns=METRIC_LABELS).round(6),
                 hide_index=True, width='stretch')

def read_batch_csv(payload):
    if len(payload) > 10 * 1024 * 1024:
        raise ValueError('Tệp vượt 10 MB.')
    content = payload.decode('utf-8-sig')
    reader = csv.reader(io.StringIO(content), strict=True)
    header = [cell.strip() for cell in next(reader, [])]
    if len(header) != len(set(header)):
        raise ValueError('Tên cột trùng nhau.')
    rows, line_numbers = [], []
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f'Dòng {reader.line_num} có số trường khác dòng tiêu đề.')
        rows.append(row)
        line_numbers.append(reader.line_num)
        if len(rows) > 10000:
            break
    if not 1 <= len(rows) <= 10000:
        raise ValueError('Cần từ 1 đến 10.000 dòng.')
    frame = pd.DataFrame(rows, columns=header)
    frame['Dòng CSV'] = line_numbers
    return frame

def remember(result,source):
    rows=result[result.prediction.notna()].copy()
    rows['timestamp_utc']=datetime.now(timezone.utc).isoformat();rows['source']=source
    st.session_state.history=(st.session_state.history+rows.to_dict('records'))[-1000:]

def explain(x):
    pipe=bundle.get('explanation_pipeline',bundle['pipeline'])
    if not hasattr(pipe,'steps') or pipe.steps[-1][1].__class__.__name__!='LGBMClassifier':
        st.info('Giải thích đóng góp hiện hỗ trợ mô hình nền LightGBM.');return
    prep,model=pipe.steps[0][1],pipe.steps[-1][1]
    features, _, _ = validate(x, {})
    z=prep.transform(features);values=np.asarray(model.booster_.predict(z,pred_contrib=True))[0]
    raw_score = float(model.booster_.predict(z, raw_score=True)[0])
    if not np.isclose(raw_score, values.sum(), atol=1e-8):
        st.warning('Chưa thể hiển thị giải thích cho mẫu này.')
        return
    names=[n.split('__')[-1] for n in prep.get_feature_names_out()]
    names=[LABELS.get(n,{'Type_M':'Loại sản phẩm M','Type_H':'Loại sản phẩm H'}.get(n,n)) for n in names]
    table=pd.DataFrame({'Thuộc tính':names,'Đóng góp log-odds':values[:-1]}).sort_values('Đóng góp log-odds')
    fig,ax=plt.subplots(figsize=(9,4));v=table['Đóng góp log-odds']
    ax.barh(table['Thuộc tính'],v,color=['#e18345' if a>0 else '#168b91' for a in v]);ax.axvline(0,color='#9baab4',lw=.8)
    ax.set_xticks([])
    ax.spines[['top','right','bottom','left']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Thanh cam hướng sang phải làm tăng điểm dự đoán hỏng. Thanh xanh hướng sang trái làm giảm điểm. Thanh dài hơn thể hiện ảnh hưởng lớn hơn trong mẫu này.')

def predict_input(frame):
    result = predict_batch(bundle, frame, meta, allow_outside=False)
    result['Kết luận'] = result.prediction.map({0: 'Chưa cần ưu tiên', 1: 'Ưu tiên kiểm tra'}).astype('string')
    result.loc[result.range_warning.ne(''), 'Kết luận'] = 'Ngoài phạm vi train — không đủ cơ sở kết luận'
    result.loc[result.input_error.ne(''), 'Kết luận'] = 'Dữ liệu không hợp lệ'
    return result

def show_result(result):
    row=result.iloc[0]
    if row.input_error:st.error(row.input_error);return
    if row.range_warning:
        st.warning('Ngoài phạm vi train — không đủ cơ sở kết luận.')
        ranges = meta['train_ranges']
        for col in NUM:
            value = float(row[col])
            lower, upper = ranges[col]['min'], ranges[col]['max']
            if not lower <= value <= upper:
                st.write(f'{LABELS[col]} · Đã nhập **{value:g}** · Khoảng huấn luyện **{lower:g} – {upper:g}**')
        return
    if row.prediction==1:st.error('MÔ HÌNH DỰ ĐOÁN: CÓ HỎNG • Nhãn 1')
    else:st.success('MÔ HÌNH DỰ ĐOÁN: KHÔNG HỎNG • Nhãn 0')
    a,b=st.columns(2);a.metric('Điểm lớp hỏng',f'{row.probability:.2%}');b.metric('Ngưỡng quyết định',f'{threshold:.2%}')
    st.caption('Điểm từ ngưỡng trở lên được đưa vào danh sách ưu tiên kiểm tra. Điểm được tính trên dữ liệu AI4I mô phỏng.')
    with st.expander('Thông số của lần dự đoán này'):st.dataframe(result[FEATURES],hide_index=True,width='stretch')
    with st.expander('Giải thích đóng góp của từng thuộc tính'):explain(result[FEATURES])
    download(result,'du_doan_mot_mau.csv')

st.markdown(f'<div class="hero"><div class="eyebrow">NHÓM 16 · PREDICTIVE MAINTENANCE</div><h1>{page}</h1><p>Phân loại trạng thái máy từ thông số vận hành. Kiểm chứng kết quả bằng dữ liệu có nhãn.</p></div>',unsafe_allow_html=True)

if page=='Tổng quan':
    a,b,c,d=st.columns(4);a.metric('Mô hình',bundle['model_name']);b.metric('Ngưỡng',f'{threshold:.2%}');c.metric('Dữ liệu AI4I','10.000 mẫu');d.metric('Đầu vào','6 thuộc tính')
    metrics=pd.read_csv(ROOT/'reports/metrics.csv');r=metrics.set_index('Split').loc['Test']
    st.subheader('Kết quả kiểm thử đã ghi nhận')
    detected, missed, false_alarm = st.columns(3)
    detected.metric('Phát hiện đúng mẫu hỏng', f'{int(r.TP)} / {int(r.TP+r.FN)}')
    missed.metric('Mẫu hỏng bị bỏ sót', int(r.FN))
    false_alarm.metric('Cảnh báo nhầm', int(r.FP))
    st.dataframe(pd.read_csv(ROOT/'reports/split_counts.csv').round(4), hide_index=True, width='stretch')
    st.caption('Đây là kết quả trên tập test của AI4I. Tập test đã được xem trong các vòng phát triển; không phải đánh giá độc lập trên máy thật.')

elif page=='Dự đoán một mẫu':
    sample=st.selectbox('Chọn dữ liệu minh họa',['Tự nhập','Mẫu không hỏng','Mẫu có hỏng'])
    presets = pd.read_csv(ROOT / 'data/mau_minh_hoa.csv').set_index('Machine failure')
    defaults = (('M', 300., 310., 1500, 40., 100) if sample == 'Tự nhập'
                else tuple(presets.loc[int(sample == 'Mẫu có hỏng'), FEATURES]))
    if st.session_state.get('last_sample')!=sample:
        st.session_state.pop('single_result',None);st.session_state.last_sample=sample
    st.caption('Hai mẫu có sẵn thuộc validation và được chọn trong các mẫu dự đoán đúng. Chúng chỉ minh họa thao tác.')
    with st.form('sample_'+sample):
        a,b=st.columns(2)
        typ=a.selectbox('Loại sản phẩm (Type)',['L','M','H'],index=['L','M','H'].index(defaults[0]))
        air=a.number_input(LABELS[NUM[0]],value=defaults[1],step=.1)
        proc=a.number_input(LABELS[NUM[1]],value=defaults[2],step=.1)
        speed=b.number_input(LABELS[NUM[2]],value=int(defaults[3]),step=1)
        torque=b.number_input(LABELS[NUM[3]],value=defaults[4],step=.1)
        wear=b.number_input('Độ mòn dụng cụ (phút sử dụng)',value=int(defaults[5]),step=1)
        submit=st.form_submit_button('Phân tích trạng thái',type='primary')
    if submit:
        x=pd.DataFrame([[typ,air,proc,speed,torque,wear]],columns=FEATURES)
        r=predict_input(x);st.session_state.single_result=r;remember(r,'single')
    if 'single_result' in st.session_state:
        st.caption('Kết quả lần bấm gần nhất. Sau khi sửa thông số, bấm Phân tích trạng thái để cập nhật.')
        show_result(st.session_state.single_result)
    with st.expander('Phạm vi quan sát trên tập train'):
        st.dataframe(pd.DataFrame(meta['train_ranges']).T.rename(columns={'min':'Nhỏ nhất','max':'Lớn nhất'}),width='stretch')
        st.caption('Khoảng quan sát của từng biến, không phải giới hạn an toàn. Ngoài khoảng sẽ tự cảnh báo; trong khoảng không bảo đảm tổ hợp hợp lý hoặc dự đoán đúng.')

elif page == 'Sàng lọc từ CSV':
    st.write('Tải các lần đo để xếp hạng ca cần kiểm tra theo điểm của mô hình.')
    a, b, c = st.columns(3)
    a.metric('Mô hình', bundle['model_name'])
    b.metric('Ngưỡng cảnh báo', f'{threshold:.2%}')
    c.metric('Chi phí bỏ sót : báo nhầm', f'{miss_cost}:{false_alarm_cost}')
    source = st.radio('Nguồn dữ liệu', ['Tệp minh họa', 'Tải CSV của bạn'], horizontal=True)
    sample_path = ROOT / 'data/mau_nhap.csv'
    download(pd.read_csv(sample_path), 'mau_nhap.csv', 'Tải CSV mẫu')
    if source == 'Tệp minh họa':
        payload = sample_path.read_bytes()
        st.caption('Tệp gồm các mẫu validation được chọn để minh họa danh sách ưu tiên. Trang Kiểm chứng test dùng toàn bộ tập test.')
    else:
        upload = st.file_uploader('Tải CSV (tối đa 10 MB và 10.000 dòng)', type='csv')
        payload = upload.getvalue() if upload else None
    fingerprint = hashlib.sha256(payload).hexdigest() if payload else None
    if st.session_state.get('upload_hash') != fingerprint:
        st.session_state.pop('batch_result', None)
        st.session_state.upload_hash = fingerprint
    if st.button('Sàng lọc ngay', type='primary'):
        st.session_state.pop('batch_result', None)
        if payload is None:
            st.warning('Hãy chọn tệp CSV trước.')
        else:
            try:
                frame = read_batch_csv(payload)
                # Recompute result fields even if a previous export is uploaded again.
                frame = frame.drop(columns=['actual', 'correct', 'Trường hợp'], errors='ignore')
                result = predict_input(frame)
                if 'Machine failure' in result:
                    result = attach_labels(result)
                st.session_state.batch_result = result
                remember(result, 'csv')
            except (ValueError, csv.Error) as exc:
                st.error(f'Không xử lý được CSV. {exc}')
    if 'batch_result' in st.session_state:
        result = st.session_state.batch_result
        alerts = result[result.prediction.eq(1)].sort_values('probability', ascending=False, kind='stable')
        a, b, c, d = st.columns(4)
        a.metric('Ưu tiên kiểm tra', len(alerts))
        b.metric('Đã dự đoán', int(result.prediction.notna().sum()))
        c.metric('Dòng sai định dạng', int(result.input_error.ne('').sum()))
        d.metric('Ngoài khoảng train', int(result.range_warning.ne('').sum()))
        queue_tab, all_tab = st.tabs(['Ưu tiên kiểm tra', 'Toàn bộ kết quả'])
        with queue_tab:
            if alerts.empty:
                st.info('Chưa có mẫu nào vượt ngưỡng cảnh báo.')
            else:
                st.dataframe(display_batch(alerts), hide_index=True, width='stretch')
                row = st.selectbox('Mở chi tiết theo dòng CSV', alerts['Dòng CSV'].tolist())
                with st.expander('Thông số và điểm của mẫu được chọn', expanded=True):
                    show_result(alerts[alerts['Dòng CSV'].eq(row)])
        with all_tab:
            if 'actual' in result:
                labeled = result.actual.notna() & result.prediction.notna()
                st.caption(f'Có {int(labeled.sum())}/{len(result)} dòng vừa có nhãn 0/1 vừa dự đoán được. Để đo hiệu năng trên tập test cố định, dùng trang Kiểm chứng test.')
            st.dataframe(display_batch(result.sort_values('probability', ascending=False, kind='stable')),
                         hide_index=True, width='stretch')
            download(result, 'ket_qua_sang_loc.csv')
            st.caption('Dòng sai định dạng hoặc ngoài phạm vi train được giữ lại kèm lý do.')
    with st.expander('Định dạng tệp CSV'):
        st.write('Giữ đúng tên sáu cột như tệp mẫu. CSV dùng dấu phẩy giữa các cột và dấu chấm thập phân. Có thể thêm Machine failure để đối chiếu.')

elif page=='Kiểm chứng test':
    st.write('Chạy mô hình và ngưỡng đã chốt trên 1.500 mẫu test, rồi đối chiếu với nhãn Machine failure.')
    test=pd.read_csv(ROOT/'data/test_goc.csv');download(test,'test_goc.csv','Tải tập test có nhãn')
    st.caption('Tập test này đã được quan sát ở các vòng phát triển trước. Việc kiểm chứng ở đây xác nhận tính nhất quán của chương trình.')
    if st.button('Chạy kiểm chứng 1.500 mẫu',type='primary'):
        r=attach_labels(predict_batch(bundle,test,meta))
        st.session_state.test_result=r
    if 'test_result' in st.session_state:
        r=st.session_state.test_result
        if r.prediction.isna().any():st.error('Có mẫu không dự đoán được. Chưa thể đánh giá toàn bộ tập test.');st.stop()
        m=measure(r.actual,r.probability,threshold)
        a,b,c,d=st.columns(4);a.metric('Precision',f'{m["Precision"]:.2%}');b.metric('Recall',f'{m["Recall"]:.2%}');c.metric('F1',f'{m["F1"]:.4f}');d.metric('PR-AUC',f'{m["PR_AUC_trapezoid"]:.4f}')
        st.write(f'Phát hiện **{m["TP"]}/{m["TP"]+m["FN"]}** mẫu hỏng, bỏ sót **{m["FN"]}**, cảnh báo nhầm **{m["FP"]}**. Chi phí quy ước ở tỷ lệ {miss_cost}:{false_alarm_cost} là **{cost_of(m, miss_cost)}**.')
        cm=pd.DataFrame([[m['TN'],m['FP']],[m['FN'],m['TP']]],index=['Thực tế không hỏng','Thực tế có hỏng'],columns=['Dự đoán không hỏng','Dự đoán có hỏng'])
        st.dataframe(cm,width='stretch')
        expected=pd.read_csv(ROOT/'reports/predictions_test.csv', float_precision='round_trip').set_index('source_row')
        ref=expected.loc[r.source_row]
        error=float(np.max(np.abs(ref.probability.to_numpy()-r.probability.to_numpy())))
        if (error < 1e-10 and np.array_equal(ref.prediction.to_numpy(), r.prediction.to_numpy())
                and np.array_equal(ref.actual.to_numpy(), r.actual.to_numpy())):
            st.success('Điểm dự đoán, nhãn dự đoán và nhãn dữ liệu khớp với lần thực nghiệm đã lưu.')
        else:
            st.error('Kết quả khác lần thực nghiệm đã lưu. Kiểm tra tệp mô hình và dữ liệu test.')
        choice=st.selectbox('Lọc trường hợp',['Tất cả'] + OUTCOMES)
        view=r if choice=='Tất cả' else r[r['Trường hợp']==choice]
        st.dataframe(display_batch(view),hide_index=True,width='stretch');download(r,'kiem_chung_test.csv')
        st.caption('Nhãn thật là nhãn có sẵn trong dataset mô phỏng, không phải kết luận từ kiểm tra máy thật.')

elif page == 'So sánh mô hình':
    st.write('So sánh 12 cấu hình trên cùng tập validation. Mô hình có AP cao nhất được chọn trước khi kiểm tra test.')
    st.caption('SMOTENC tạo mẫu lớp hỏng trong train. Validation và test giữ tỷ lệ gốc.')
    experiments, simulator, evidence = st.tabs(['Xử lý mất cân bằng', 'Ngưỡng theo chi phí', 'Kết quả đã chốt'])
    with experiments:
        comparison = pd.read_csv(ROOT / 'reports/validation_comparison.csv')
        alg = st.selectbox('Thuật toán', ['Tất cả'] + comparison['Algorithm'].drop_duplicates().tolist())
        visible = comparison if alg == 'Tất cả' else comparison[comparison.Algorithm.eq(alg)]
        columns = ['Model', 'Fit normal', 'Fit failure', 'AP', 'PR_AUC_trapezoid',
                   'F1', 'TP', 'FN', 'FP', 'Cost']
        st.dataframe(visible[columns].rename(columns=METRIC_LABELS).round(4),
                     hide_index=True, width='stretch')
        st.caption(f'F1 và số ca trong bảng dùng ngưỡng tối thiểu chi phí riêng của mỗi cấu hình trên validation ở tỷ lệ {miss_cost}:{false_alarm_cost}. AP và PR-AUC tổng hợp qua nhiều ngưỡng.')
        download(comparison, 'so_sanh_validation.csv', 'Tải bảng đầy đủ')
    with simulator:
        st.write('Thử thay đổi chi phí bỏ sót để xem số ca bỏ sót và báo nhầm trên validation.')
        scenario = st.number_input('Chi phí bỏ sót một ca hỏng, tính theo số lượt kiểm tra nhầm',
                                   min_value=1, value=miss_cost, step=1)
        val = pd.read_csv(ROOT / 'reports/predictions_validation.csv', float_precision='round_trip')
        curve = threshold_curve(val.actual, val.probability)
        cuts = select_thresholds(curve, scenario)
        options = []
        for policy in ('Cost', 'F1'):
            metrics = measure(val.actual, val.probability, cuts[policy])
            options.append({'Policy': POLICIES[policy], 'Threshold': cuts[policy], **metrics,
                            'Cost': cost_of(metrics, scenario)})
        show_policies(pd.DataFrame(options))
        chart = curve.assign(**{'Chi phí quy ước': cost_of(curve, scenario)})
        st.scatter_chart(chart, x='Threshold', y='Chi phí quy ước',
                         x_label='Ngưỡng trên validation', height=240)
        st.caption('Kịch bản chi phí là giả định của nhóm. Thao tác thử ngưỡng ở đây chỉ tác động đến bảng validation.')
    with evidence:
        st.caption(f'Kịch bản chính thức của demo dùng chi phí bỏ sót và báo nhầm theo tỷ lệ {miss_cost}:{false_alarm_cost}.')
        policies = pd.read_csv(ROOT / 'reports/policy_comparison.csv')
        split = st.radio('Tập đánh giá', ['Test', 'Validation'], horizontal=True)
        view = policies.loc[policies.Split.eq(split)].copy()
        view.Policy = view.Policy.map(POLICIES)
        view.loc[view.Policy.eq(POLICIES['All normal']), 'Threshold'] = np.nan
        show_policies(view)
        download(policies, 'so_sanh_nguong.csv', 'Tải kết quả các ngưỡng đã chốt')
        st.image(str(ROOT / 'reports/pr_curve.png'), width=600)

elif page == 'Đọc ảnh bảng đo':
    st.write('Tải ảnh chụp bảng có sáu thông số. Kiểm tra và sửa lại giá trị đã đọc trước khi dự đoán.')
    sample_image = ROOT / 'data/mau_bang_do.png'
    st.download_button('Tải ảnh bảng đo mẫu', sample_image.read_bytes(), sample_image.name, 'image/png')
    source = st.radio('Nguồn ảnh', ['Ảnh mẫu', 'Tải ảnh của bạn'], horizontal=True)
    upload = st.file_uploader('Ảnh JPG hoặc PNG, tối đa 5 MB', type=['jpg', 'jpeg', 'png']) if source == 'Tải ảnh của bạn' else None
    payload = sample_image.read_bytes() if source == 'Ảnh mẫu' else (upload.getvalue() if upload else None)
    if payload:
        fingerprint = hashlib.sha256(payload).hexdigest()
        if st.session_state.get('image_hash') != fingerprint:
            st.session_state.image_hash = fingerprint
            st.session_state.pop('ocr_fields', None)
            st.session_state.pop('ocr_result', None)
        if len(payload) > 5 * 1024 * 1024:
            st.error('Ảnh vượt 5 MB.')
            st.stop()
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in ('JPEG', 'PNG') or image.width * image.height > 12_000_000:
                    raise ValueError('Chọn ảnh JPG hoặc PNG tối đa 12 megapixel.')
                st.image(image, caption='Ảnh bảng đo đã chọn', width=440)
        except (UnidentifiedImageError, Image.DecompressionBombError, ValueError, OSError) as exc:
            st.error(f'Ảnh không hợp lệ. {exc}')
            st.stop()
        if st.button('Đọc thông số', type='primary'):
            st.session_state.pop('ocr_fields', None)
            st.session_state.pop('ocr_result', None)
            try:
                st.session_state.ocr_fields = read_sensor_image(payload)
            except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
                st.error(f'Chưa đọc được ảnh. {exc}')
        if 'ocr_fields' in st.session_state:
            fields = st.session_state.ocr_fields
            if not all(fields.values()):
                st.warning('Một số ô còn trống. Điền từ ảnh hoặc tải ảnh rõ hơn.')
            with st.form('ocr_review_' + fingerprint):
                values = {key: st.text_input(LABELS.get(key, 'Loại sản phẩm L, M hoặc H'), fields[key])
                          for key in FEATURES}
                confirmed = st.checkbox('Tôi đã đối chiếu sáu giá trị với ảnh')
                submit = st.form_submit_button('Dự đoán từ thông số đã xác nhận', type='primary')
            if submit:
                st.session_state.pop('ocr_result', None)
                if confirmed:
                    values = {key: value.strip().replace(',', '.') for key, value in values.items()}
                    result = predict_input(pd.DataFrame([values]))
                    st.session_state.ocr_result = result
                    remember(result, 'ảnh bảng đo')
                else:
                    st.warning('Hãy xác nhận đã đối chiếu thông số trước khi dự đoán.')
            if 'ocr_result' in st.session_state:
                show_result(st.session_state.ocr_result)
    st.caption('OCR đọc chữ và số theo mẫu TYPE, AIR K, PROCESS K, SPEED RPM, TORQUE NM, WEAR MIN. Kết quả dự đoán dùng các giá trị đã xác nhận.')

elif page=='Lịch sử':
    st.caption('Lưu tối đa 1.000 dự đoán gần nhất của phiên. Đóng phiên hoặc khởi động lại có thể mất lịch sử; tải CSV để lưu lâu dài. Kiểm chứng test không ghi vào lịch sử.')
    if st.button('Xóa lịch sử phiên'):st.session_state.history=[]
    hist=pd.DataFrame(st.session_state.history)
    if hist.empty:st.info('Chưa có dự đoán trong phiên này.')
    else:st.dataframe(hist,hide_index=True,width='stretch');download(hist,'lich_su_du_doan.csv')

else:
    st.subheader('Dữ liệu và phạm vi bài toán')
    st.write('AI4I 2020 có 10.000 mẫu mô phỏng, trong đó 339 mẫu mang nhãn hỏng. Nhóm chia phân tầng thành 7.000 mẫu train, 1.500 mẫu validation và 1.500 mẫu test.')
    st.link_button('Nguồn dữ liệu UCI','https://archive.ics.uci.edu/dataset/601/ai4i')
    st.dataframe(pd.DataFrame({'Cột':['Type']+NUM,'Ý nghĩa':['Loại sản phẩm L/M/H']+list(LABELS.values())}),hide_index=True,width='stretch')
    st.write('Sáu đầu vào được mã hóa thành bảy đặc trưng: năm biến số và hai cột Type_M, Type_H; loại L là nhóm tham chiếu. Nhiệt độ dùng Kelvin, không nhập trực tiếp độ C.')
    st.write('Đầu vào loại trừ UDI, Product ID và các cột kiểu hỏng TWF, HDF, PWF, OSF, RNF. Machine failure là nhãn dùng để huấn luyện và đối chiếu.')
    st.subheader('Ý nghĩa của kết quả')
    st.write('Nhãn 1 biểu thị mẫu được dự đoán hỏng, nhãn 0 biểu thị mẫu được dự đoán không hỏng. Nhóm dùng nhãn Machine failure trong dữ liệu để đánh giá kết quả.')
    st.write('Khi thông số tự nhập hoặc trong CSV vượt khoảng đã quan sát trên train, ứng dụng báo chưa đủ cơ sở kết luận. Phần kiểm chứng thực nghiệm đối chiếu toàn bộ tập test đã cố định với nhãn dữ liệu.')
    st.caption('Phạm vi nghiên cứu là phân loại trạng thái trên AI4I 2020. Thời điểm hỏng và tình trạng của máy thật cần dữ liệu kiểm chứng riêng.')

st.markdown('<div class="footer">NHÓM 16 · HỌC MÁY &nbsp; / &nbsp; AI4I 2020 · Demo nghiên cứu trên dữ liệu mô phỏng</div>',unsafe_allow_html=True)
