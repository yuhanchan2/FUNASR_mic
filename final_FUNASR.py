import pip
pip.main(['install', 'torch', 'torchaudio', 'pyaudio', 'numpy'])
pip.main(['install', 'funasr'])
pip.main(['install', 'modelscope', 'huggingface', 'huggingface_hub'])
import pyaudio
import numpy as np
from funasr import AutoModel

# Model parameters (same as in your code)
chunk_size = [0, 10, 5]
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1
frames_per_chunk = chunk_size[1] * 960  # 600ms chunk at 16kHz
channels = 1
rate = 16000

# Initialize model and cache
model = AutoModel(model="paraformer-zh-streaming")
cache = {}

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,  # use Int16 for WAV-like PCM
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=frames_per_chunk)

print("Listening... (Press Ctrl+C to stop)")
try:
    while True:
        data = stream.read(frames_per_chunk)
        # Convert byte data to numpy array
        speech_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0  # normalize if needed
        # Feed chunk into the model
        res = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=False,  # Set True if it's the last chunk
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back
        )
        print(res)
except KeyboardInterrupt:
    print("Stopped recording.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()