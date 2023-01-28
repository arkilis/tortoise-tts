#!/c/Users/Ben/anaconda3/envs/tortoise-tts/python
import uuid
import os
import time
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# start time
start_time = time.time()

# This will download all the models used by Tortoise from the HuggingFace hub.

useCPU = False
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not useCPU) else 'cpu')
print('Using device:', DEVICE)
if not useCPU:
    if torch.cuda.get_device_capability(DEVICE) == (8,0): ## A100 fix thanks to Emad
        print('Disabling CUDNN for A100 gpu', file=sys.stderr)
        torch.backends.cudnn.enabled = False

tts = TextToSpeech(device=DEVICE)

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "standard"
#voice = 'train_dotrice'
voice = 'bedtimestory1'

# Load it and send it through Tortoise.
voice_samples, conditioning_latents = load_voice(voice)


def generate(file_name, text):
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)
    torchaudio.save(file_name, gen.squeeze(0).cpu(), 24000)
    print("--- {} seconds: {} ---".format("%0.2f" % (time.time() - start_time), file_name))


SOURCE_PATH = "stories"
SOURCE_FILE = "source.txt"
def generate_stories():
    lines = [] # final lines will be spoken

    dirs = os.listdir(os.path.join(os.getcwd(), SOURCE_PATH))
    for dir in dirs:
        dir_full_path = os.path.join(os.getcwd(), SOURCE_PATH, dir)
        if os.path.isdir(dir_full_path):
            file_full_path = os.path.join(dir_full_path, SOURCE_FILE)
            if os.path.exists(file_full_path) and "done_" not in file_full_path:
                print("reading story from dir [" + dir + " ]...")
                lines = open(file_full_path).readlines()
                lines = list( map(lambda x:x.strip(), lines))
                lines = [x for x in lines if x]
                # print(lines)

                
                file_name = str(uuid.uuid4())
                for index, line in enumerate(lines):
                    voice_file = os.path.join(dir_full_path, file_name+"_"+str(index)+".wav")
                    print("saved file name: " + voice_file)
                    generate(voice_file, line)

generate_stories()