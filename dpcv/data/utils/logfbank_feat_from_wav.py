from python_speech_features import logfbank
import scipy.io.wavfile as wav


def process_wav(wav_file):
    (rate, sig) = wav.read(wav_file)
    fbank_feat = logfbank(sig, rate)  # fbank_feat.shape = (3059,26)
    a = fbank_feat.flatten()
    single_vec_feat = a.reshape(1, -1)  # single_vec_feat.shape = (1,79534)
    return single_vec_feat


if __name__ == "__main__":
    process_wav("/home/rongfan/05-personality_traits/DeepPersonality/datasets/demo/-6otZ7M-Mro.003.wav")