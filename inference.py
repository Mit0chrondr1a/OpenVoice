# pylint: disable=invalid-name
import os

import se_extractor
from api import ToneColorConverter

device = "cuda:0"
output_dir = "outputs"

ckpt_base = "checkpoints/base_speakers"
ckpt_converter = "./checkpoints/converter"
language_option = "ZH"

# init model
tone_color_converter = ToneColorConverter(
    f"{ckpt_converter}/config.json", device=device
)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
os.makedirs(output_dir, exist_ok=True)


# use prebuilt speaker encoder or OPENAI's speaker encoder
# prebuilt speaker encoder simply load pth file, while OPENAI's needs se_extractor.get_se

base_speaker = f"{output_dir}/台词_openai_output.mp3"
source_se, audio_name = se_extractor.get_se(
    base_speaker, tone_color_converter, vad=True
)

# base_speaker_tts = BaseSpeakerTTS(
#     f"{ckpt_base}/{language_option}/config.json", device=device
# )
# base_speaker_tts.load_ckpt(f"{ckpt_base}/{language_option}/checkpoint.pth")

# source_se = torch.load(f"{ckpt_base}/{language_option}/zh_default_se.pth").to(device)


reference_speaker = "resources/dada.wav"
target_se, audio_name = se_extractor.get_se(
    reference_speaker, tone_color_converter, target_dir="processed", vad=True
)

moodForOutput = "default"
save_path = f"{output_dir}/Chinese/output_{moodForOutput}.wav"

text = "何公子,今天天气真好，我们一起出去吃饭吧"
src_path = f"{output_dir}/tmp.wav"
# base_speaker_tts.tts(
#     text, src_path, speaker=moodForOutput, language="Chinese", speed=1.0
# )

# inference
encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path=src_path,
    src_se=source_se,
    tgt_se=target_se,
    output_path=save_path,
    message=encode_message,
)
