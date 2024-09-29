import json
import os

from Components.FaceCrop import crop_to_vertical, combine_videos
from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video
from Components.Transcription import transcribeAudio, convert_to_srt_file, burn_subtitles
from Components.DynamicDetection import calculate_dynamics_brightness
from Components.ClipsHighlighter import search_peaks
from Components.EmotionSentiment import EmotionSentiment
# from Components.FaceCrop import crop_to_vertical, combine_videos



if __name__ == "__main__":
    path = "videos/video2.mp4"
    Vid = download_youtube_video(path)
    if Vid:
        #Vid = Vid.replace(".webm", ".mp4")
        print(f"Downloaded video and audio files successfully! at {Vid}")

        Audio = extractAudio(path)
        if Audio:

            transcriptions = transcribeAudio(Audio)
            dynamics_list = calculate_dynamics_brightness(path)
            sentiment = EmotionSentiment()
            sentiments = [[sentiment.predict_emotion(transcription[0]),transcription[1], transcription[2]] for transcription in transcriptions]
            intervals, _ = search_peaks(dynamics_list, transcriptions, sentiments,verbose=True)
            if len(intervals) > 0:
                for start, stop in intervals:
                    if start != 0 and stop != 0:
                        print(f"Start: {start} , End: {stop}")

                        Output = f"Out{start}.mp4"

                        crop_video(path, Output, start, stop)
                        cropped = f"croped{start}.mp4"

                        crop_to_vertical(f"Out{start}.mp4", cropped)
                        combine_videos(f"Out{start}.mp4", cropped, f"Final{start}.mp4")
                        Audio = extractAudio(f"Final{start}.mp4")
                        transcriptions = transcribeAudio(Audio)
                        srt_file = convert_to_srt_file(transcriptions)
                        burn_subtitles(srt_file.name, f"Final{start}.mp4", f"SubFinal{start}.mp4")
                        os.remove(Output)
                        os.remove(f"Final{start}.mp4")


            else:
                print("No transcriptions found")
        else:
            print("No audio file found")
    else:
        print("Unable to Download the video")