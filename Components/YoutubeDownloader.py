
import cv2

def get_video_size(stream):

    return stream.filesize / (1024 * 1024)

def download_youtube_video(url):
    try:
        video_file = cv2.VideoCapture(url)
        return video_file

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure you have the latest version of pytube and ffmpeg-python installed.")
        print("You can update them by running:")
        print("pip install --upgrade pytube ffmpeg-python")
        print("Also, ensure that ffmpeg is installed on your system and available in your PATH.")

if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    download_youtube_video(youtube_url)
