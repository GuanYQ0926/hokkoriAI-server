from pydub import AudioSegment


def set_to_target_level(sound, target_level):
    difference = target_level - sound.dBFS
    return sound.apply_gain(difference)


cry_sound = AudioSegment.from_file()
sing_sound = AudioSegment.from_file()

combined = cry_sound.overlay(sing_sound)
combined.export('', format='wav')
