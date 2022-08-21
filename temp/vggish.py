# import torch
#
# model = torch.hub.load('harritaylor/torchvggish', 'vggish')
# model.eval()
#
# # Download an example audio file
# import urllib
# url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)
#
# model.forward(filename)
from torchvggish import vggish, vggish_input

# Initialise model and download weights
embedding_model = vggish()
embedding_model.eval()
# example = vggish_input.wavfile_to_examples("/home/rongfan/05-personality_traits/vggish/bus_chatter.wav")
# example = vggish_input.wavfile_to_examples("/home/rongfan/05-personality_traits/DeepPersonality/datasets/chalearn2021/test/talk_test/008105/FC1_T.wav")
example = vggish_input.wavfile_to_examples("/media/rongfan/EXTERNAL_USB/DeepPersonalityData/raw_data/raw_audio/validationData/-6otZ7M-Mro.003.wav")
embeddings = embedding_model.forward(example)
print(embeddings.shape)
