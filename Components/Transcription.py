import subprocess
import tempfile



def burn_subtitles(srt_path, input_video, output_file):
    # Команда FFmpeg в виде списка аргументов
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=10,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3,Outline=1,Shadow=1,Alignment=2,MarginV=10'",
        '-c:a', 'copy',
        output_file,
        '-y'
    ]

    try:
        # Запуск команды FFmpeg
        subprocess.run(ffmpeg_command, check=True)
        print("Subtitles burned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def convert_to_srt_file(segments):
    """
    Конвертирует список сегментов в формат SRT и записывает во временный файл.

    :param segments: Список сегментов в формате [["text", start, end], ...]
    :return: Объект временного файла
    """

    def format_time(seconds):
        # Форматирует время в формате SRT (чч:мм:сс,мс)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

    srt_output = []

    for index, (text, start, end) in enumerate(segments, start=1):
        start_time = format_time(start)
        end_time = format_time(end)

        # Форматирование каждого сегмента в стиле SRT
        srt_output.append(f"{index}")
        srt_output.append(f"{start_time} --> {end_time}")
        srt_output.append(text)
        srt_output.append("")  # Пустая строка между записями

    # Создание временного файла
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.srt', encoding='utf-8')

    # Запись содержимого в файл
    temp_file.write("\n".join(srt_output))
    temp_file.flush()  # Сбрасываем буфер на диск

    # Возвращаем указатель файла на начало
    temp_file.seek(0)

    return temp_file

def transcribeAudio(audio_path, model):
    try:

        prompt = "Please ensure complete sentences are transcribed without truncation."
        segments, info = model.transcribe(audio=audio_path , initial_prompt=prompt)#, beam_size=5, language="ru", max_new_tokens=128, condition_on_previous_text=False)


        segments = list(segments)
        # print(segments)
        extracted_texts = [[segment.text, segment.start, segment.end] for segment in segments]
        return extracted_texts
    except Exception as e:
        print("Transcription Error:", e)
        return []

if __name__ == "__main__":
    audio_path = "audio.wav"
    transcriptions = transcribeAudio(audio_path)
    print("Done")
    TransText = ""

    for text, start, end in transcriptions:
        TransText += f"{start} - {end}: {text}"
    print(TransText)