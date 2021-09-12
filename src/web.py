import os
import random
import tempfile
import traceback
import dataclasses
from urllib.request import urlopen

import cv2 as cv
import numpy as np
import mediapipe as mp
import streamlit as st
import pandas as pd


# [start]____________________________________________________________
# [start]____________________________________________________________
# [strings.py]

pageConfig = """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 500px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 500px;
            margin-left: -500px;
        }
        </style>
        """
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

nbsp = "&nbsp"
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]\




# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
#
#
#
# [start]____________________________________________________________
# [start]____________________________________________________________
# [modules.py]

# [start]____________________________________________________________
class handDetector:
    def __init__(self, imageMode=False, numHands=2, solutionConfidence=0.5):
        self.imageMode = imageMode
        self.numHands = numHands
        self.detectionConfidence = solutionConfidence
        self.trackingConfidence = solutionConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.imageMode,
            self.numHands,
            self.detectionConfidence,
            self.trackingConfidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findFeatures(self, imgRGB):
        self.results = self.hands.process(imgRGB)
        ih, iw = imgRGB.shape[:2]

        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(imgRGB, handLM, self.mpHands.HAND_CONNECTIONS)

                for idx, LM in enumerate(handLM.landmark):
                    x_pixels, y_pixels = int(LM.x * iw), int(LM.y * ih)
                    if idx in [0, 4, 8, 12, 16, 20]:
                        cv.circle(
                            imgRGB, (x_pixels, y_pixels), 5, (255, 0, 255), cv.FILLED
                        )
        
        self.img = imgRGB 
        self.data = self.results 
        return self


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start]____________________________________________________________
class poseDetector:
    def __init__(self, imageMode=False, smoothLandmarks=True, solutionConfidence=0.5):
        self.imageMode = imageMode
        self.smoothLandmarks = smoothLandmarks
        self.detectionConfidence = solutionConfidence
        self.trackingConfidence = solutionConfidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.imageMode,
            self.smoothLandmarks,
            self.detectionConfidence,
            self.trackingConfidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findFeatures(self, imgRGB):
        self.results = self.pose.process(imgRGB)
        ih, iw = imgRGB.shape[:2]

        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                imgRGB, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
            )

            for idx, LM in enumerate(self.results.pose_landmarks.landmark):
                x_pixels, y_pixels = int(LM.x * iw), int(LM.y * ih)
                cv.circle(imgRGB, (x_pixels, y_pixels), 5, (255, 0, 0), cv.FILLED)
                
        self.img = imgRGB 
        self.data = self.results 
        return self

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start]____________________________________________________________
class faceDetector:
    def __init__(self, solutionConfidence=0.5, modelSelection=1):
        self.detectionConfidence = solutionConfidence
        self.modelSelection = modelSelection

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(
            self.detectionConfidence, self.modelSelection
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findFeatures(self, imgRGB):
        ih, iw = imgRGB.shape[:2]
        self.results = self.face.process(imgRGB)

        if self.results.detections:
            for idx, detection in enumerate(self.results.detections):
                mp_bbox = detection.location_data.relative_bounding_box
                bbox = (
                    int(mp_bbox.xmin * iw),
                    int(mp_bbox.ymin * ih),
                    int(mp_bbox.width * iw),
                    int(mp_bbox.height * ih),
                )
                score = round(detection.score[-1] * 100, ndigits=2)

                imgRGB = self.fancyDraw(imgRGB, bbox, score)

        self.img = imgRGB 
        self.data = self.results 
        return self

    def fancyDraw(self, img, bbox, score, thick_rect=2):
        x0, y0, w, h = bbox
        x1, y1 = x0 + w, y0 + h
        len_line = int(0.25 * w)
        thick_line = thick_rect + 3

        # bounding box
        cv.rectangle(img, bbox, (99, 19, 247), thick_rect)
        # border - top left: x0, y0
        cv.line(img, (x0, y0), (x0 + len_line, y0), (99, 19, 247), thick_line)
        cv.line(img, (x0, y0), (x0, y0 + len_line), (99, 19, 247), thick_line)
        # border - top right: x1, y0
        cv.line(img, (x1, y0), (x1 - len_line, y0), (99, 19, 247), thick_line)
        cv.line(img, (x1, y0), (x1, y0 + len_line), (99, 19, 247), thick_line)
        # border - bottom left: x0, y1
        cv.line(img, (x0, y1), (x0 + len_line, y1), (99, 19, 247), thick_line)
        cv.line(img, (x0, y1), (x0, y1 - len_line), (99, 19, 247), thick_line)
        # border - bottom left: x1, y1
        cv.line(img, (x1, y1), (x1 - len_line, y1), (99, 19, 247), thick_line)
        cv.line(img, (x1, y1), (x1, y1 - len_line), (99, 19, 247), thick_line)
        # score text
        cv.putText(
            img,
            f"{score}%",
            (bbox[0], bbox[1] - 8),
            cv.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        return img


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start]____________________________________________________________
class faceMeshDetector:
    def __init__(self, imageMode=False, numFaces=1, solutionConfidence=0.5):
        self.imageMode = imageMode
        self.numFaces = numFaces
        self.detectionConfidence = solutionConfidence
        self.trackingConfidence = solutionConfidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.imageMode,
            self.numFaces,
            self.detectionConfidence,
            self.trackingConfidence,
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.landmark_drawSpecs = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 251, 251)
        )
        self.connection_drawSpecs = self.mpDraw.DrawingSpec(
            thickness=2, circle_radius=2, color=(251, 0, 251)
        )

    def findFeatures(self, imgRGB):
        ih, iw = imgRGB.shape[:2]
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for idx, faceLM in enumerate(self.results.multi_face_landmarks):
                self.mpDraw.draw_landmarks(
                    imgRGB,
                    faceLM,
                    self.mpFaceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.landmark_drawSpecs,
                    connection_drawing_spec=self.connection_drawSpecs,
                )

        self.img = imgRGB 
        self.data = self.results 
        return self


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
#
#
#
# [start]____________________________________________________________
# [start]____________________________________________________________
# [functions.py]

# [start] [defaults] ________________________________________________
# local
dataPath = r"C:\\Users\\MSI\Desktop\\autistomat\\data"
demoImages = ["reshot01.jpg", "reshot02.jpg", "reshot03.jpg", "reshot04.jpg"]
demoVideos = ["pexels03.mp4", "pexels04.mp4", "pexels05.mp4", "pexels08.mp4"]
demoWebCam = "webcam_image.png"
# online
urlImages = [
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838655/photosp/0a0f136f-9032-4480-a1ca-1185dd161368/0a0f136f-9032-4480-a1ca-1185dd161368.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838912/photosp/57ece171-00ea-439c-95e2-01523fd41285/57ece171-00ea-439c-95e2-01523fd41285.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838985/photosp/fc11c636-e5db-4b02-b3d2-99650128c351/fc11c636-e5db-4b02-b3d2-99650128c351.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838983/photosp/22f4d64b-6358-47b5-8a96-a5b9b8019829/22f4d64b-6358-47b5-8a96-a5b9b8019829.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1613176985/photosp/373e53b2-49a8-4e5f-85d2-bfc1a233572e/373e53b2-49a8-4e5f-85d2-bfc1a233572e.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1588713749/photosp/148fb738-110b-45d5-96e5-89bd36335b91/148fb738-110b-45d5-96e5-89bd36335b91.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1541429083/photosp/76400249-2c7c-4030-b381-95c1c4106db6/76400249-2c7c-4030-b381-95c1c4106db6.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1525791591/photosp/ee9ec8f6-3f11-48a6-a112-57825f983b3a/ee9ec8f6-3f11-48a6-a112-57825f983b3a.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838947/photosp/705bc78c-6e43-47c1-8295-802b48106695/705bc78c-6e43-47c1-8295-802b48106695.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1588713965/photosp/4c7d6a68-a215-47bd-a1a6-7fb137cdf6c4/4c7d6a68-a215-47bd-a1a6-7fb137cdf6c4.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1611254271/photosp/c67f2181-50a6-4b4d-9099-d401197a99a2/c67f2181-50a6-4b4d-9099-d401197a99a2.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1623132298/photosp/fafc9d9c-f2dc-46f5-9fa4-885914b176b0/fafc9d9c-f2dc-46f5-9fa4-885914b176b0.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838971/photosp/a62f2758-ca17-424f-b406-d646162202d7/a62f2758-ca17-424f-b406-d646162202d7.jpg",
]
urlVideos = [
    "https://assets.mixkit.co/videos/preview/mixkit-joyful-lovers-hugging-4643-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-friends-with-colored-smoke-bombs-4556-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-group-of-friends-partying-happily-4640-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-silhouette-of-a-couple-jumping-in-the-sunset-33417-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-girl-dancing-happy-at-home-8752-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-multiracial-people-joining-hands-23012-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-a-woman-and-a-man-in-a-beautiful-park-4882-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-mother-with-her-happy-daughters-4549-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-excited-girl-talking-on-video-call-with-her-cell-phone-8745-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-excited-young-people-partying-with-party-hats-4608-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-young-woman-talking-on-video-call-on-a-terrace-10442-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-happy-smiling-students-promote-college-education-9001-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-woman-doing-mountain-climber-exercise-726-large.mp4",
    "https://assets.mixkit.co/videos/preview/mixkit-man-training-on-the-bars-in-the-gym-23450-large.mp4",
]
# global parameters
target_h, target_w = 350, 550

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start] [persistent states]__________________________________________
@dataclasses.dataclass
class functionsState:
    current_image_path: str = demoImages[0]
    current_image_url: str = urlImages[0]
    idx_url_image: int = 0
    current_video_path: str = demoVideos[0]
    current_video_url: str = urlVideos[0]
    idx_url_video: int = 0
    sol_confidence: float = 0.65
    num_hands: int = 2
    smooth_lms: int = 1
    face_model: int = 0
    num_faces: int = 2
    current_image_upload: str = ""
    current_video_upload: str = ""
    uploader_key: int = 0


@st.cache(allow_output_mutation=True)
def _functionsState() -> functionsState:
    return functionsState()


_fs = _functionsState()

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


def open_img_path_url(url_or_file, source_type, source_path=None, resize=False):
    img, mask = [], []

    if source_type == "path":
        if source_path is None:
            source_path = dataPath
        img = cv.imread(os.path.join(source_path, url_or_file))

    elif source_type == "url":
        resp = urlopen(url_or_file)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv.imdecode(img, cv.IMREAD_COLOR)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if not resize:
        return img

    else:
        img_h = img.shape[0]
        ratio = target_h / img_h
        r_img = cv.resize(img, None, fx=ratio, fy=ratio)

        r_img_w = r_img.shape[1]
        left_edge = target_w // 2 - r_img_w // 2

        mask = np.zeros((target_h, target_w, 3), dtype="uint8")
        mask[:, left_edge : left_edge + r_img_w] = r_img

        return img, mask


def open_vid_path_url(url_or_file, source_type, source_path=None, preview=False):
    vid, vid_preview = [], []

    if source_type == "path":
        if source_path is None:
            source_path = dataPath
        vid_preview = os.path.join(source_path, url_or_file)

    elif source_type == "url":
        vid_preview = url_or_file

    vid = cv.VideoCapture(vid_preview)

    if preview:
        return vid, vid_preview
    else:
        return vid


def open_webcam():
    vid = cv.VideoCapture(0)
    vid.set(3, 1280)  # width
    vid.set(4, 720)  # height

    return vid


def read_source_media(data_source_selection):
    if data_source_selection == "User Image":
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        img_file_buffer = st.sidebar.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"], key=_fs.uploader_key
        )

        if img_file_buffer:
            temp_file.write(img_file_buffer.read())
            _fs.current_image_upload = temp_file.name

        if _fs.current_image_upload != "":
            img, mask = open_img_path_url(
                _fs.current_image_upload, "path", source_path="", resize=True
            )

            st.sidebar.markdown("")
            cols = st.sidebar.columns([2, 1])
            cols[0].text("Original Image")
            st.sidebar.image(mask, use_column_width=True)
            if cols[1].button("Clear Upload"):
                _fs.current_image_upload = ""
                _fs.uploader_key += 1
                st.experimental_rerun()
            st.sidebar.markdown("---")

            return img, "image"

    elif data_source_selection == "Random Local Image":
        img = open_img_path_url(_fs.current_image_path, "path")

        st.sidebar.markdown("")
        cols = st.sidebar.columns([2, 1])
        cols[0].text("Original Image")
        st.sidebar.image(img, use_column_width=True)
        if cols[1].button("Change Image"):
            _fs.current_image_path = random.choice(demoImages)
            st.experimental_rerun()
        st.sidebar.markdown("---")

        return img, "image"

    elif data_source_selection == "Random Online Image":
        img = open_img_path_url(_fs.current_image_url, "url")

        st.sidebar.markdown("")
        cols = st.sidebar.columns([2, 1])
        cols[0].text("Original Image")
        st.sidebar.image(img, use_column_width=True)
        if cols[1].button("Change Image"):
            _fs.idx_url_image += 1
            _fs.current_image_url = urlImages[_fs.idx_url_image % len(urlImages)]
            st.experimental_rerun()
        st.sidebar.markdown("---")

        return img, "image"

    elif data_source_selection == "User Video":
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        vid_file_buffer = st.sidebar.file_uploader(
            "Upload Video", type=["mp4", "mov", "avi", "3gp", "m4v"], key=_fs.uploader_key
        )

        if vid_file_buffer:
            temp_file.write(vid_file_buffer.read())
            _fs.current_video_upload = temp_file.name

        if _fs.current_video_upload != "":
            vid = open_vid_path_url(_fs.current_video_upload, "path")

            st.sidebar.markdown("")
            cols = st.sidebar.columns([2, 1])
            cols[0].text("Original Video")
            st.sidebar.video(_fs.current_video_upload)
            if cols[1].button("Clear Upload"):
                _fs.current_video_upload = ""
                _fs.uploader_key += 1
                st.experimental_rerun()
            st.sidebar.markdown("---")

            return vid, "video"

    elif data_source_selection == "Random Local Video":
        vid, vid_preview = open_vid_path_url(_fs.current_video_path, "path", preview=True)

        st.sidebar.markdown("")
        cols = st.sidebar.columns([2, 1])
        cols[0].text("Original Video")
        st.sidebar.video(vid_preview)
        if cols[1].button("Change Video"):
            _fs.current_video_path = random.choice(demoVideos)
            st.experimental_rerun()
        st.sidebar.markdown("---")

        return vid, "video"

    elif data_source_selection == "Random Online Video":
        vid, vid_preview = open_vid_path_url(_fs.current_video_url, "url", preview=True)

        st.sidebar.markdown("")
        cols = st.sidebar.columns([2, 1])
        cols[0].text("Original Video")
        st.sidebar.video(vid_preview)
        if cols[1].button("Change Video"):
            _fs.idx_url_video += 1
            _fs.current_video_url = urlVideos[_fs.idx_url_video % len(urlVideos)]
            st.experimental_rerun()
        st.sidebar.markdown("---")

        return vid, "video"

    elif data_source_selection == "WebCam":
        vid = open_webcam()

        st.sidebar.text("WebCam")
        st.sidebar.image(open_img_path_url(demoWebCam, "path"))
        st.sidebar.markdown("---")

        return vid, "webcam"

    return None, None  # final fallback


def init_module(media, type, detector, placeholders):
    frame_count = 0
    cols = placeholders[0].columns([2, 2, 1, 1])

    if type == "image":
        img = detector.findFeatures(media)
        placeholders[1].image(img, use_column_width=True)

    if type in ["video", "webcam"]:
        stop_clicked = cols[0].button("🟥 STOP")
        start_clicked = cols[3].button("🟢 START")
        chart_data = pd.DataFrame( [0], columns=['a'])
        chart = st.line_chart(chart_data)
        cycle=10
        cntchart=0
        while True:
            try:
                success, img = media.read()
                frame_count += 1
                
                if type == "webcam":
                    img = cv.flip(img, 1)

                if frame_count == media.get(cv.CAP_PROP_FRAME_COUNT):
                    frame_count = 0
                    media.set(cv.CAP_PROP_POS_FRAMES, 0)

                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                data = detector.findFeatures(img)
                img = data.img
                placeholders[1].image(img, use_column_width=True)
                
                if (cntchart>cycle):
                    cntchart=0
                    newpoint = 0
                    if(hasattr(data.data, 'multi_hand_landmarks') and data.data.multi_hand_landmarks) :
                        for handLM in data.data.multi_hand_landmarks:
                            
                            for idx, LM in enumerate(handLM.landmark):
                                #x_pixels, y_pixels = int(LM.x ), int(LM.y )
                                if idx in [8]:
                                    ih, iw = img.shape[:2]
                                    newpoint =int( ih -( LM.y * ih))
                    chart_data = pd.DataFrame( [newpoint ], columns=['a'])
                    chart.add_rows(chart_data)         
                        
                cntchart +=1
                
                        
                        

                if stop_clicked:
                    placeholders[1].empty()
                    media.release()
                    cv.destroyAllWindows()
                    break

                if start_clicked:
                    st.experimental_rerun()

            except Exception:
                placeholders[1].info(traceback.format_exc())
                media.release()
                cv.destroyAllWindows()
                break


def run_selected_module(module_selection, media, type, ph_variables):
    moreInfo1 = st.empty()
    moreInfo2 = st.empty()
    moduleOutput1 = st.empty()
    moduleOutput2 = st.empty()

    _fs.sol_confidence = ph_variables[0].slider(
        "Solution Confidence [0.4-1.0]",
        min_value=0.4,
        max_value=1.0,
        value=_fs.sol_confidence,
    )

    if module_selection == "Hand Tracking":
        
        new_value = ph_variables[1].number_input(
            "Number Of Hands [1-6]", min_value=1, max_value=6, value=_fs.num_hands
        )
        if new_value != _fs.num_hands:
            _fs.num_hands = new_value
            st.experimental_rerun()

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = handDetector(
            numHands=_fs.num_hands, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Pose Estimation":
        
        new_value = ph_variables[1].number_input(
            "Smooth Landmarks [0/1]", min_value=0, max_value=1, value=_fs.smooth_lms
        )
        if new_value != _fs.smooth_lms:
            _fs.smooth_lms = new_value
            st.experimental_rerun()

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = poseDetector(
            smoothLandmarks=bool(_fs.smooth_lms), solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Face Detection":
        
        new_value = ph_variables[1].number_input(
            "Model Selection [0/1]", min_value=0, max_value=1, value=_fs.face_model
        )
        if new_value != _fs.face_model:
            _fs.face_model = new_value
            st.experimental_rerun()

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = faceDetector(
            modelSelection=_fs.face_model, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Face Mesh":
        
        new_value = ph_variables[1].number_input(
            "Number Of Faces [1-5]", min_value=1, max_value=5, value=_fs.num_faces
        )
        if new_value != _fs.num_faces:
            _fs.num_faces = new_value
            st.experimental_rerun()

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = faceMeshDetector(
            numFaces=_fs.num_faces, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
#
#
#
# [start]____________________________________________________________
# [start]____________________________________________________________
# [streamlitMediapipe.py]

# [start] [persistent states]__________________________________________
@dataclasses.dataclass
class webappState:
    idx_current_page: int = 1
    current_page: str = "About Web App"
    page_selector_key: int = 0

    idx_current_module: int = 0
    current_module: str = "Hand Tracking"
    module_selector_key: int = 0

    idx_data_source: int = 1
    data_source: str = "Random Image"
    source_selector_key: int = 0


@st.cache(allow_output_mutation=True)
def _webappState() -> webappState:
    return webappState()


webapp = _webappState()
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start] [setup main page and side bar] ____________________________
st.markdown(
    f"""
    {pageConfig}
    <div style="text-align:center; margin-top:-75px; ">
    <h1 style="font-variant: small-caps; font-size: xx-large; margin-bottom:-45px;" >
    <font color=#ea0525>A U T I S T O M A T</font>
    </h1>
    <hr>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div style="text-align:center; margin-top:-75px; ">
    <h3 style="font-variant: small-caps; font-size: xx-large; ">
    <font color=#ea0525>s e t t i n g s</font>
    </h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

# [start] [setup app pages, modules & data sources]__________________

appPages = ["Mediapipe Modules", "About Web App", "About Me"]
page_selection = "Mediapipe Modules"
# st.sidebar.selectbox(
#    "Page Selection:",
#    appPages,
#    index=webapp.idx_current_page,
#)
if webapp.current_page != page_selection:
    webapp.idx_current_page = appPages.index(page_selection)
    webapp.current_page = page_selection
    st.experimental_rerun()

webapp.current_page = "Mediapipe Modules"
if webapp.current_page == "About Me":
    st.sidebar.markdown("---")
    st.sidebar.image(open_img_path_url("highlord.jpg", "path"), use_column_width="auto")
    st.sidebar.markdown("---")



elif webapp.current_page == "Mediapipe Modules":
    st.set_option("deprecation.showfileUploaderEncoding", False)

    mp_selectors = st.sidebar.columns([1, 1])

    appModules = ["Hand Tracking", "Pose Estimation", "Face Detection", "Face Mesh"]
    module_selection = mp_selectors[0].selectbox(
        "Choose The Mediapipe Solution:",
        appModules,
        index=webapp.idx_current_module,
    )
    if webapp.current_module != module_selection:
        webapp.idx_current_module = appModules.index(module_selection)
        webapp.current_module = module_selection
        st.experimental_rerun()

    appDataSources = [
        "User Video",
        "WebCam"
    ]
    data_source_selection = mp_selectors[1].selectbox(
        "Select Media Source:",
        appDataSources,
        index=webapp.idx_data_source,
    )
    if webapp.data_source != data_source_selection:
        webapp.idx_data_source = appDataSources.index(data_source_selection)
        webapp.data_source = data_source_selection
        st.experimental_rerun()

    st.sidebar.markdown("")
    ph_variables = st.sidebar.columns([1, 1])
    st.sidebar.markdown("")

    media, type = read_source_media(webapp.data_source)
    run_selected_module(webapp.current_module, media, type, ph_variables)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]
st.markdown(""" <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """, unsafe_allow_html=True)