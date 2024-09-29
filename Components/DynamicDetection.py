import cv2

def calculate_dynamics_brightness(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * 10)

    dynamics_list = []
    interval_dynamics_score = 0
    frame_count = 0

    ret, prev_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр.")
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        frame_dynamics_score = diff.mean()
        interval_dynamics_score += frame_dynamics_score

        prev_gray = gray
        frame_count += 1

        if frame_count % interval_frames == 0:
            average_dynamics = interval_dynamics_score / interval_frames
            start_time = (frame_count - interval_frames) / fps
            end_time = frame_count / fps
            dynamics_list.append([average_dynamics, start_time, end_time])
            interval_dynamics_score = 0

    if frame_count % interval_frames != 0:
        average_dynamics = interval_dynamics_score / (frame_count % interval_frames)
        start_time = (frame_count - (frame_count % interval_frames)) / fps
        end_time = frame_count / fps
        dynamics_list.append([average_dynamics, start_time, end_time])

    cap.release()
    return dynamics_list