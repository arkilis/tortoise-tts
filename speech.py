import time
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
import uuid

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

# Here's something for the poetically inclined.. (set text=)
text = """
Once upon a time, in a faraway land, there lived a kind and gentle king who ruled over a beautiful and prosperous kingdom. The king was known for his wisdom and fairness, and his people loved and respected him.
<break>
One day, the king decided to go on a journey to visit all the corners of his kingdom and make sure that everyone was happy and healthy. He traveled from town to town, from village to village, and from farm to farm, listening to the concerns of his people and doing everything he could to help them.
<break>
As the king was traveling, he came across a small cottage in the woods. The cottage was old and run-down, and it looked as if no one had lived there for a very long time. But as the king approached the door, he heard a faint sound coming from inside.
<break>
He knocked on the door and called out, Hello, is anyone there?
<break>
To his surprise, a tiny voice answered, Yes, Your Majesty. Please come in.
<break>
The king opened the door and entered the cottage. Inside, he found a small, old woman sitting by the fireplace. She was so tiny that she barely reached up to the king's knee.
<break>
How long have you lived here all by yourself? the king asked the woman.
<break>
Oh, I don't know, the woman replied. It feels like forever. I'm all alone, and I don't have anyone to talk to or take care of me.
<break>
The king felt sorry for the woman, and he decided to help her. He ordered his men to fix up the cottage and make it warm and comfortable. He also sent some of his best cooks to prepare delicious meals for the woman and make sure she had everything she needed.
<break>
The woman was overjoyed by the king's kindness. She thanked him and promised to be forever grateful.
<break>
The king continued his journey and visited many more people in need. He helped the sick, the poor, and the lonely, and everywhere he went, he was greeted with smiles and thanks.
<break>
Finally, after many months, the king returned to his palace. He was tired but happy, knowing that he had made a difference in the lives of his people.
<break>
And from that day on, the king was known not only for his wisdom and fairness, but also for his kindness and generosity. His kingdom flourished, and everyone lived happily ever after.
"""

def generate(file_name, text):
    # Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
    preset = "standard"
    #voice = 'train_dotrice'
    voice = 'bedtimestory1'

    # Load it and send it through Tortoise.
    voice_samples, conditioning_latents = load_voice(voice)
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                            preset=preset)
    torchaudio.save(file_name, gen.squeeze(0).cpu(), 24000)
    print("--- {} seconds: {} ---".format("%0.2f" % (time.time() - start_time), file_name))

file_name = str(uuid.uuid4())
for index, p in enumerate(text.split("<break>")):
    generate(file_name+"_"+str(index)+".wav", p)