from collections import defaultdict

from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np

def smooth_data(data, window_size=3):
    """Функция для сглаживания данных с помощью скользящего среднего."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
def normalize_data(data):
    """Функция для нормализации данных."""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def calculate_peak_intervals(y_smooth, peaks):
    # Вычисляем ширину пиков на полувысоте
    widths_half_max = peak_widths(y_smooth, peaks, rel_height=0.5)
    return widths_half_max


def create_intervals(peaks, widths_half_max):
    intervals = []
    max_interval_length = 180

    for i, peak in enumerate(peaks):
        start_time = max(0, int(widths_half_max[2][i]))
        end_time = int(widths_half_max[3][i])
        if end_time - start_time > max_interval_length:
            end_time = start_time + max_interval_length

        intervals.append([start_time, end_time])

    return intervals


def merge_intervals(intervals, overlap_threshold=0.45):
    # Сортируем интервалы по началу
    intervals.sort(key=lambda x: x[0])
    filtered_intervals = []

    for interval in intervals:
        if not filtered_intervals:
            filtered_intervals.append(interval)
        else:
            last_interval = filtered_intervals[-1]
            overlap = min(last_interval[1], interval[1]) - max(last_interval[0], interval[0])
            if overlap > 0:
                overlap_ratio = overlap / (last_interval[1] - last_interval[0])
                if overlap_ratio > overlap_threshold:
                    if (interval[1] - interval[0]) > (last_interval[1] - last_interval[0]):

                        filtered_intervals[-1] = interval
                else:
                    filtered_intervals.append(interval)
            else:
                filtered_intervals.append(interval)

    return filtered_intervals

def expand_dynamics_to_seconds(dynamics_values, interval_length=10):
    """Расширяет каждый элемент массива на заданное количество секунд."""
    expanded_values = []
    for value in dynamics_values:
        expanded_values.extend([value] * interval_length)
    return expanded_values
def expand_transcriptions_to_seconds(transcriptions):
    """Расширяет текст на каждую секунду в зависимости от интервала."""
    expanded_transcriptions = []
    for text, start, stop in transcriptions:
        duration = int(stop - start)
        expanded_transcriptions.extend([(text, second) for second in range(int(start), int(stop))])
    return expanded_transcriptions

def calculate_activity_for_intervals(merged_intervals, y_smooth):
    activities = []

    for interval in merged_intervals:
        start, stop = interval
        if start < 0 or stop >= len(y_smooth) or start > stop:
            # Либо индекс вне диапазона, либо интервал некорректен
            continue

        activity = sum(y_smooth[start:stop + 1])
        activities.append(activity)

    if not activities:
        return []

    min_activity = min(activities)
    max_activity = max(activities)

    if max_activity == min_activity:
        normalized_activities = [50] * len(activities)
    else:
        normalized_activities = [
            100 * (activity - min_activity) / (max_activity - min_activity)
            for activity in activities
        ]

    return normalized_activities

def search_peaks(dynamics_list, transcriptions, sentiments, verbose=False):
    dynamics_values = [d[0] for d in dynamics_list]
    start_seconds = [d[1] for d in dynamics_list]

    smoothed_dynamics_values = smooth_data(expand_dynamics_to_seconds(dynamics_values))
    normalized_dynamics_values = normalize_data(smoothed_dynamics_values)
    if verbose:
        plt.figure(figsize=(14, 6))
        plt.plot(range(len(normalized_dynamics_values)), normalized_dynamics_values, marker='o')


        plt.title("График динамичности (сглаженный и нормализованный)")
        plt.xlabel("Начальная секунда интервала")
        plt.ylabel("Нормализованная динамичность")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("example_plot.png", format='png')
        plt.clf()

    word_count_per_interval = defaultdict(set)
    expanded_transcriptions = expand_transcriptions_to_seconds(transcriptions)
    for text, second in expanded_transcriptions:
        word_count_per_interval[second].update(text)

    intervals = sorted(word_count_per_interval.keys())
    unique_word_counts = [len(word_count_per_interval[i]) for i in intervals]
    smoothed_unique_word_counts = smooth_data(unique_word_counts)
    normalized_unique_word_counts = normalize_data(smoothed_unique_word_counts)

    if verbose:
        plt.figure(figsize=(14, 6))
        plt.plot(range(len(normalized_unique_word_counts)), normalized_unique_word_counts, marker='o')

        plt.title("График количества уникальных слов (сглаженный и нормализованный)")
        plt.xlabel("Начальная секунда интервала")
        plt.ylabel("Нормализованное количество уникальных слов")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("example_plot2.png", format='png')
        plt.clf()

    # Добавление интонаций

    expanded_sentiment = expand_transcriptions_to_seconds(sentiments)
    sentiment_values = [val[0] for val in expanded_sentiment ]
    sentiment_values = smooth_data(sentiment_values)
    normalized_sentiment_values = normalize_data(sentiment_values)

    if verbose:
        plt.figure(figsize=(14, 6))
        plt.plot(range(len(normalized_sentiment_values)), normalized_sentiment_values, marker='o')


        plt.title("График интонации (сглаженный и нормализованный)")
        plt.xlabel("Начальная секунда интервала")
        plt.ylabel("Нормализованная интонация")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("example_plot3.png", format='png')
        plt.clf()

    # Сложение графиков
    min_length = min(len(normalized_dynamics_values), len(normalized_unique_word_counts), len(normalized_sentiment_values))
    combined_values = (normalized_dynamics_values[:min_length] + normalized_unique_word_counts[:min_length] +
                       normalized_sentiment_values[:min_length])
    smooth_combined_value = smooth_data(combined_values, max(int(len(combined_values) * 0.05), 3))
    x_original = np.arange(len(smooth_combined_value))
    f_interp = PchipInterpolator(x_original, smooth_combined_value)
    x_smooth = np.linspace(x_original.min(), x_original.max(), len(x_original))
    y_smooth = f_interp(x_smooth)
    # Поиск пиков
    peaks, _ = find_peaks(y_smooth, width=20)
    peaks = peaks[peaks < len(y_smooth)]
    widths_half_max = calculate_peak_intervals(y_smooth, peaks)
    intervals = create_intervals(peaks, widths_half_max)
    merged_intervals = merge_intervals(intervals)

    if verbose:
        # Построение результирующего графика с пиками
        plt.figure(figsize=(14, 6))
        plt.plot(x_smooth, y_smooth, marker='o', label='Сложенный график')
        plt.plot(peaks, y_smooth[peaks], "x", label='Пики')

        plt.title("Сложенный график с пиками")
        plt.xlabel("Начальная секунда интервала")
        plt.ylabel("Нормализованное значение")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("combined_plot_with_peaks.png", format='png')
        plt.clf()
    return merged_intervals, calculate_activity_for_intervals(merged_intervals, y_smooth)
