#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1k0EEHiTj_oLUbfr3oFY5c0bm4Plk4uX0'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
#st.cache_resource

def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 음식의 이름")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 음식의 성분    음식의 레시피")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(2):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)
    # 2nd Row - YouTube Videos
    for i in range(2):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(2):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/DrXkGH6/52007-52699-1320.jpg",
            "https://i.ibb.co/gv5pwPx/image.png"
        ],
        'videos': [
            "https://youtu.be/k7Cg3w_-FFM?feature=shared",
            "https://youtu.be/QNipXG5LPik?feature=shared"
        ],
        'texts': [
            "떡볶이에는 밀떡 또는 쌀떡, 파, 고추장, 마늘, 양파, 고춧가루 등이 들어갑니다.",
            "맛있는 떡볶이 레시피"
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/RGnS8xy/20190221173302-8.png",
            "https://i.ibb.co/hc15H7Z/1.jpg"
        ],
        'videos': [
            "https://youtu.be/KxVArmhyN2U?feature=shared",
            "https://youtu.be/mJKVKTLN1CA?feature=shared"
        ],
        'texts': [
            "어묵은 주로 연육으로 만들어지는데, 당근이나 양파 등이 들어가는 어묵도 있습니다.",
            "맛있는 분식집 어묵꼬지 레시피"
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/PDpkQc7/60398052.jpg",
            "https://i.ibb.co/GWmpMKj/um-I-he-VYVS9mi-QNq-XM13-FRUOHHL4l1nzs-Zg-N9-XRLFG7n-I-7-Dyf-Myr6-Hmi-Wf9-Qd7-SAZQz3-WYSQHPXXt-GAw-L.webp"
        ],
        'videos': [
            "https://youtu.be/imemPs_j4gQ?feature=shared",
            "https://youtu.be/jxVOPjOTKoo?feature=shared"
        ],
        'texts': [
            "피자는 밀가루 반죽인 피자 도우, 베이컨과 토마토 소스, 각종 야채, 올리브, 피자의 종류에 따른 다양한 토핑이 들어갑니다.",
            "맛있는 피자 레시피"
        ]
    },
     labels[3]: {
        'images': [
            "hhttps://i.ibb.co/cvGXXft/output-2369372993.jpg",
            "https://i.ibb.co/9qBdZK5/410-0023-0.jpg"
        ],
        'videos': [
            "https://youtu.be/QZ7W1Er0-Gs?feature=shared",
            "https://youtu.be/MWlHuWmnyek?feature=shared"
        ],
        'texts': [
            "후라이드 치킨은 주재료인 닭, 튀김가루, 소금 등이 들어갑니다.",
            "맛있는 후라이드 치킨 레시피"
        ]
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

