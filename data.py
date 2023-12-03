from pydub import AudioSegment


def split_and_save_wav(input_file, output_folder):
    # Load the WAV file
    audio = AudioSegment.from_wav(input_file)

    # Calculate the duration of each segment in milliseconds (20 seconds)
    segment_duration = 5 * 1000

    # Iterate over segments and save them with specific names
    for i, start_time in enumerate(range(0, len(audio), segment_duration)):
        segment = audio[start_time:start_time + segment_duration]
        output_file = f"{output_folder}/custom-{i}-X-50.wav"
        segment.export(output_file, format="wav")
        print(f"Segment {i + 1} saved as {output_file}")


# Example usage
input_wav_file = "silence.wav"
output_folder_path = "data/ESC-50-master/audio"
split_and_save_wav(input_wav_file, output_folder_path)
