import hashlib
import os
import tempfile
import threading
import time
from os.path import join, dirname
from queue import Queue
from flask import request,jsonify, send_file
import torch
import requests
from tqdm import tqdm

from Components.EmotionSentiment import EmotionSentiment
from Components.FaceCrop import combine_videos, crop_to_vertical
from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video
from Components.Transcription import transcribeAudio, convert_to_srt_file, burn_subtitles
from Components.DynamicDetection import calculate_dynamics_brightness
from Components.ClipsHighlighter import search_peaks
from faster_whisper import WhisperModel
import torch
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
DOMEN = os.environ.get("domen")

from app import app

def md5_checksum(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def run_application():
    if __name__ == '__main__':
        threading.Thread(target=lambda: app.run(debug=False)).start()


class Task:
    def __init__(self, task_type, func, kwargs: dict[str, any]):
        self.task_id = hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()
        self.data = {"id": self.task_id, "type": task_type, "status": "process", "content": {}}
        self.func = func
        self.kwargs = kwargs
        self.status_code = 200

    def set_result(self, content, status_code):
        if content is None:
            self.data["content"] = {}
            self.data["status"] = "abort"
            self.status_code = status_code
        else:
            self.data["status"] = "ready"
            self.data["content"] = content
            self.status_code = status_code

def process_intervals(intervals, source,  model):
    temp_files_final = []
    if len(intervals) > 0:
        for start, stop in intervals:
            if start != 0 and stop != 0:
                print(f"Start: {start} , End: {stop}")

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_out:
                    Output = temp_out.name

                crop_video(source, Output, start, stop)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_cropped:
                    cropped = temp_cropped.name
                crop_to_vertical(Output, cropped)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_final:
                    final_output = temp_final.name
                combine_videos(Output, cropped, final_output)

                Audio = extractAudio(final_output)
                transcriptions = transcribeAudio(Audio, model)
                srt_file = convert_to_srt_file(transcriptions)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_final_subs:
                    final_subs_output = temp_final_subs.name

                burn_subtitles(srt_file.name, final_output, final_subs_output)

                temp_out.close()
                temp_cropped.close()
                temp_final.close()
                temp_files_final.append([os.path.basename(final_subs_output), transcriptions])
        return temp_files_final

class MainApplication:
    def __init__(self):
        self.sentiment = EmotionSentiment()
        Device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu")
        self.queue = Queue()
        self.cache = {}

    def start(self):
        while True:
            if self.queue.empty():
                continue
            task = self.queue.get_nowait()
            print(f"task: {task.data}" )
            if not task.data.get("status") == "process":
                continue
            content, status_code = task.func(task.kwargs)
            task.set_result(content, status_code)
            self.cache[task.data["id"]] = task

    def add_task(self, task: Task):
        self.queue.put(task)
        self.cache[task.data["id"]] = task

    def get_clips_boundaries(self, filename):
        if filename:
            # Vid = Vid.replace(".webm", ".mp4")
            print(f"Downloaded video and audio files successfully! at {filename}")

            Audio = extractAudio(filename)
            if Audio:

                transcriptions = transcribeAudio(Audio, self.whisper_model)
                dynamics_list = calculate_dynamics_brightness(filename)
                sentiments = [[self.sentiment.predict_emotion(transcription[0]), transcription[1], transcription[2]] for
                              transcription in transcriptions]
                intervals, viralities = search_peaks(dynamics_list, transcriptions, sentiments, verbose=False)
                if len(transcriptions) > 0:
                    print(f"Найдено {len(intervals)} клипов")
                    files = process_intervals(intervals, filename, self.whisper_model)
                    return {
                        "highlights": [
                            {
                                "start": start,
                                "end": end,
                                "virality": virality,
                                "transcription": [
                                    {
                                        "text": t[0],
                                        "start": t[1],
                                        "end": t[2]
                                    }
                                    for t in transcription
                                ],
                                "file": f"{DOMEN}/download/{file}"
                            }
                            for (start, end), (file, transcription), virality in zip(intervals, files, viralities)
                        ]
                    }

                else:
                    print("No transcriptions found")
                    return None
            else:
                print("No audio file found")
                return None
        else:
            print("Unable to Download the video")
            return None




application = MainApplication()


def get_clips_boundaries_wrapper(kwargs):
    result = application.get_clips_boundaries(kwargs.get("filename"))
    kwargs.get("temp_dir").cleanup()
    if not result:
        return {"error": "error while highlighting video"}, 500
    return result, 200


@app.route('/set_video_download', methods=['POST'])
def set_video_download():
    temp_dir = tempfile.TemporaryDirectory()
    download_url = request.json.get("download_url")
    filename = request.json.get("filename")
    hashsum_md5 = request.json.get("md5")
    if download_url == "":
        return jsonify({"error": "download url cant be empty"}), 422

    if filename == "":
        return jsonify({"error": "filename cant be empty"}), 422

    response = requests.head(download_url)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 МБ
    filename = download_url.split("/")[-1]
    response = requests.get(download_url, stream=True)
    with open(os.path.join(temp_dir.name, filename), "wb") as handle:
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
            for data in response.iter_content(block_size):
                handle.write(data)
                progress_bar.update(len(data))
    response.close()
    # hashsum = md5_checksum(os.path.join(temp_dir.name, filename))
    # if hashsum != hashsum_md5:
    #     return jsonify({"error": "file integrity cant be verified"}), 500
    task = Task("index_in_db", get_clips_boundaries_wrapper, {"temp_dir": temp_dir,
                                                       "filename": os.path.join(temp_dir.name, filename)})
    application.add_task(task)
    return jsonify({"task_id": task.task_id})


@app.route('/task_status', methods=['POST'])
def task_status():
    task_id = request.json.get('task_id')
    if task_id is None or task_id == "" or task_id not in application.cache.keys():
        return jsonify({"result": "error", "error_str": "task_id cant be empty"}), 422
    return jsonify(application.cache[task_id].data["content"]), application.cache[task_id].status_code

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = f"/tmp/{filename}"
    return send_file(file_path)

run_application()
application.start()