
from moviepy.editor import *
from Components.Speaker import *

global Fps
# Update paths to the model files
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
temp_audio_path = "temp_audio.wav"

# Load DNN model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness mode from 0 to 3
global Frames
Frames = [] # [x,y,w,h]

def voice_activity_detection(audio_frame, sample_rate=16000):
    return vad.is_speech(audio_frame, sample_rate)

def extract_audio_from_video(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")

def process_audio_frame(audio_data, sample_rate=16000, frame_duration_ms=30):
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    offset = 0
    while offset + n <= len(audio_data):
        frame = audio_data[offset:offset + n]
        offset += n
        yield frame



def detect_faces_and_speakers(input_video_path, output_video_path):
    # Extract audio from the video
    extract_audio_from_video(input_video_path, temp_audio_path)

    # Read the extracted audio
    with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    (x, y ,x1, y1) = (None, None, None, None)
    frame_duration_ms = 30  # 30ms frames
    audio_generator = process_audio_frame(audio_data, sample_rate, frame_duration_ms)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        audio_frame = next(audio_generator, None)
        if audio_frame is None:
            break
        is_speaking_audio = voice_activity_detection(audio_frame, sample_rate)
        MaxDif = 0
        Add = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_width = x1 - x
                face_height = y1 - y

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                # Assuming lips are approximately at the bottom third of the face
                lip_distance = abs((y + 2 * face_height // 3) - (y1))
                Add.append([[x, y, x1, y1], lip_distance])

                MaxDif == max(lip_distance, MaxDif)
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_width = x1 - x
                face_height = y1 - y

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                # Assuming lips are approximately at the bottom third of the face
                lip_distance = abs((y + 2 * face_height // 3) - y1)

                # Combine visual and audio cues
                if lip_distance >= MaxDif and is_speaking_audio:  # Adjust the threshold as needed
                    cv2.putText(frame, "Active Speaker", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if lip_distance >= MaxDif:
                    break
        if x is None:
            continue
        Frames.append([x, y, x1, y1])

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    os.remove(temp_audio_path)


def crop_to_vertical(input_video_path, output_video_path):
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vertical_height = int(original_height)
    vertical_width = int(vertical_height * 9 / 16)
    print(vertical_height, vertical_width)

    if original_width < vertical_width:
        print("Error: Original video width is less than the desired vertical width.")
        return

    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    print(f"start and end - {x_start} , {x_end}")
    print(x_end - x_start)
    half_width = vertical_width // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    global Fps
    Fps = fps
    print(fps)
    count = 0
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > -1:
            (x, y, w, h) = (None, None, None, None)
            if len(faces) == 0:
                if count < len(Frames):
                    (x, y, w, h) = Frames[count]

            # (x, y, w, h) = faces[0]
            try:
                # check if face 1 is active
                (X, Y, W, H) = Frames[count]
            except Exception as e:
                if count < len(Frames):
                    (X, Y, W, H) = Frames[count][0]

            for f in faces:
                x1, y1, w1, h1 = f
                center = x1 + w1 // 2
                if center > X and center < X + W:
                    x = x1
                    y = y1
                    w = w1
                    h = h1
                    break

            # print(faces[0])
            if x is not None:
                centerX = x + (w // 2)
                if count == 0 or (x_start - (centerX - half_width)) < 1:
                    ## IF dif from prev fram is low then no movement is done
                    pass  # use prev vals
                else:
                    x_start = centerX - half_width
                    x_end = centerX + half_width

                    if int(cropped_frame.shape[1]) != x_end - x_start:
                        if x_end < original_width:
                            x_end += int(cropped_frame.shape[1]) - (x_end - x_start)
                            if x_end > original_width:
                                x_start -= int(cropped_frame.shape[1]) - (x_end - x_start)
                        else:
                            x_start -= int(cropped_frame.shape[1]) - (x_end - x_start)
                            if x_start < 0:
                                x_end += int(cropped_frame.shape[1]) - (x_end - x_start)
                        #print("Frame size inconsistant")

        count += 1
        cropped_frame = frame[:, x_start:x_end]
        if cropped_frame.shape[1] == 0:
            x_start = (original_width - vertical_width) // 2
            x_end = x_start + vertical_width
            cropped_frame = frame[:, x_start:x_end]


        out.write(cropped_frame)

    cap.release()
    out.release()
    print("Cropping complete. The video has been saved to", output_video_path, count)


def combine_videos(video_with_audio, video_without_audio, output_filename):
    try:
        # Load video clips
        clip_with_audio = VideoFileClip(video_with_audio)
        clip_without_audio = VideoFileClip(video_without_audio)

        audio = clip_with_audio.audio

        combined_clip = clip_without_audio.set_audio(audio)

        global Fps
        combined_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac', fps=Fps, preset='medium',
                                      bitrate='3000k')
        print(f"Combined video saved successfully as {output_filename}")

    except Exception as e:
        print(f"Error combining video and audio: {str(e)}")


