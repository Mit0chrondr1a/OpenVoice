# pylint: disable=invalid-name
import argparse
import os

import gradio as gr
import langid
import torch

import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--share", action="store_true", default=False, help="make link public"
)
args = parser.parse_args()

en_ckpt_base = "checkpoints/base_speakers/EN"
zh_ckpt_base = "checkpoints/base_speakers/ZH"
ckpt_converter = "checkpoints/converter"
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "outputs"
input_dir = "inputs"
os.makedirs(output_dir, exist_ok=True)

# load models
en_base_speaker_tts = BaseSpeakerTTS(f"{en_ckpt_base}/config.json", device=device)
en_base_speaker_tts.load_ckpt(f"{en_ckpt_base}/checkpoint.pth")
zh_base_speaker_tts = BaseSpeakerTTS(f"{zh_ckpt_base}/config.json", device=device)
zh_base_speaker_tts.load_ckpt(f"{zh_ckpt_base}/checkpoint.pth")
tone_color_converter = ToneColorConverter(
    f"{ckpt_converter}/config.json", device=device
)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

# load speaker embeddings
en_source_default_se = torch.load(f"{en_ckpt_base}/en_default_se.pth").to(device)
en_source_style_se = torch.load(f"{en_ckpt_base}/en_style_se.pth").to(device)

# base_speaker = f"{input_dir}/OPENAI_TTS.mp3"
# zh_source_se, cn_audio_name = se_extractor.get_se(
#     base_speaker, tone_color_converter, vad=True
# )

# Original example uses zh_default_se.pth
zh_source_se = torch.load(f"{zh_ckpt_base}/zh_default_se.pth").to(device)

# This online demo mainly supports English and Chinese
supported_languages = ["zh", "en"]


def predict(prompt, style, audio_file_pth, agree):
    # initialize a empty info
    text_hint = ""
    # # agree with the terms
    # if agree is False:
    #     text_hint += '[ERROR] Please accept the Terms & Condition!\n'
    #     gr.Warning("Please accept the Terms & Condition!")
    #     return (
    #         text_hint,
    #         None,
    #         None,
    #     )

    # first detect the input language
    language_predicted = langid.classify(prompt)[0].strip()
    print(f"Detected language:{language_predicted}")

    if language_predicted not in supported_languages:
        text_hint += f"[ERROR] The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}\n"
        gr.Warning(
            f"The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}"
        )

        return (
            text_hint,
            None,
            None,
        )

    if language_predicted == "zh":
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        language = "Chinese"
        if style not in ["default"]:
            text_hint += (
                f"[ERROR] The style {style} is not supported for Chinese, "
                "which should be in ['default']\n"
            )
            gr.Warning(
                "The style {style} is not supported for Chinese, "
                "which should be in ['default']"
            )
            return (
                text_hint,
                None,
                None,
            )

    else:
        tts_model = en_base_speaker_tts
        if style == "default":
            source_se = en_source_default_se
        else:
            source_se = en_source_style_se
        language = "English"
        if style not in [
            "default",
            "whispering",
            "shouting",
            "excited",
            "cheerful",
            "terrified",
            "angry",
            "sad",
            "friendly",
        ]:
            text_hint += f"[ERROR] The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']\n"
            gr.Warning(
                f"The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']"
            )
            return (
                text_hint,
                None,
                None,
            )

    sampleWavToBeImitated = audio_file_pth

    if len(prompt) < 2:
        text_hint += "[ERROR] Please give a longer prompt text \n"
        gr.Warning("Please give a longer prompt text")
        return (
            text_hint,
            None,
            None,
        )
    # if len(prompt) > 200:
    #     text_hint += "[ERROR] Text length limited to 200 characters for this demo,"
    #     "please try shorter text. You can clone our open-source repo and try for your usage \n"
    #     gr.Warning(
    #         "Text length limited to 200 characters for this demo, please try shorter text."
    #         "You can clone our open-source repo for your usage"
    #     )
    #     return (
    #         text_hint,
    #         None,
    #         None,
    #     )

    # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
    try:
<<<<<<< HEAD
        target_se, audio_name = se_extractor.get_se(
            sampleWavToBeImitated, tone_color_converter, target_dir="processed", vad=True
        )
||||||| parent of acd97c0 (Modify to my useage pattern)
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
=======
        target_se, audio_name = se_extractor.get_se(
            speaker_wav, tone_color_converter, target_dir="processed", vad=True
        )
>>>>>>> acd97c0 (Modify to my useage pattern)
    except Exception as e:
        text_hint += f"[ERROR] Get target tone color error {str(e)} \n"
        gr.Warning("[ERROR] Get target tone color error {str(e)} \n")
        return (
            text_hint,
            None,
            None,
        )

    src_path = f"{output_dir}/tmp.wav"
    tts_model.tts(prompt, src_path, speaker=style, language=language)

<<<<<<< HEAD
    save_path = f"{output_dir}/output.wav"
||||||| parent of acd97c0 (Modify to my useage pattern)
    save_path = f'{output_dir}/output.wav'
    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)
=======
    save_path = f"{output_dir}/output.wav"
    # Run the tone color converter
    # encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        # message=encode_message,
    )
>>>>>>> acd97c0 (Modify to my useage pattern)

<<<<<<< HEAD
    # Run the tone color converter
    # encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        # message=encode_message,
    )

    text_hint += """Get response successfully \n"""
||||||| parent of acd97c0 (Modify to my useage pattern)
    text_hint += f'''Get response successfully \n'''
=======
    text_hint += """Get response successfully \n"""
>>>>>>> acd97c0 (Modify to my useage pattern)

    return (
        text_hint,
        save_path,
        sampleWavToBeImitated,
    )

    # Skip toning shit
    # return (
    #     text_hint,
    #     src_path,
    #     sampleWavToBeImitated,
    # )

# title = "MyShell OpenVoice"

<<<<<<< HEAD
# title = "MyShell OpenVoice"
||||||| parent of acd97c0 (Modify to my useage pattern)
title = "MyShell OpenVoice"
=======
# description = """
# We introduce OpenVoice, a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages. OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. OpenVoice also achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set.
# """
>>>>>>> acd97c0 (Modify to my useage pattern)

<<<<<<< HEAD
# description = """
# We introduce OpenVoice, a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages. OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. OpenVoice also achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set.
# """
||||||| parent of acd97c0 (Modify to my useage pattern)
description = """
We introduce OpenVoice, a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages. OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. OpenVoice also achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set.
"""
=======
# markdown_table = """
# <div align="center" style="margin-bottom: 10px;">
>>>>>>> acd97c0 (Modify to my useage pattern)

<<<<<<< HEAD
# markdown_table = """
# <div align="center" style="margin-bottom: 10px;">
||||||| parent of acd97c0 (Modify to my useage pattern)
markdown_table = """
<div align="center" style="margin-bottom: 10px;">
=======
# |               |               |               |
# | :-----------: | :-----------: | :-----------: |
# | **OpenSource Repo** | **Project Page** | **Join the Community** |
# | <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> | [OpenVoice](https://research.myshell.ai/open-voice) | [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |
>>>>>>> acd97c0 (Modify to my useage pattern)

<<<<<<< HEAD
# |               |               |               |
# | :-----------: | :-----------: | :-----------: |
# | **OpenSource Repo** | **Project Page** | **Join the Community** |
# | <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> | [OpenVoice](https://research.myshell.ai/open-voice) | [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |
||||||| parent of acd97c0 (Modify to my useage pattern)
|               |               |               |
| :-----------: | :-----------: | :-----------: | 
| **OpenSource Repo** | **Project Page** | **Join the Community** |        
| <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> | [OpenVoice](https://research.myshell.ai/open-voice) | [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |
=======
# </div>
# """
>>>>>>> acd97c0 (Modify to my useage pattern)

<<<<<<< HEAD
# </div>
# """
||||||| parent of acd97c0 (Modify to my useage pattern)
</div>
"""
=======
# markdown_table_v2 = """
# <div align="center" style="margin-bottom: 2px;">
>>>>>>> acd97c0 (Modify to my useage pattern)

<<<<<<< HEAD
# markdown_table_v2 = """
# <div align="center" style="margin-bottom: 2px;">
||||||| parent of acd97c0 (Modify to my useage pattern)
markdown_table_v2 = """
<div align="center" style="margin-bottom: 2px;">
=======
# |               |               |               |              |
# | :-----------: | :-----------: | :-----------: | :-----------: |
# | **OpenSource Repo** | <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> |  **Project Page** |  [OpenVoice](https://research.myshell.ai/open-voice) |
>>>>>>> acd97c0 (Modify to my useage pattern)

<<<<<<< HEAD
# |               |               |               |              |
# | :-----------: | :-----------: | :-----------: | :-----------: |
# | **OpenSource Repo** | <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> |  **Project Page** |  [OpenVoice](https://research.myshell.ai/open-voice) |
||||||| parent of acd97c0 (Modify to my useage pattern)
|               |               |               |              |
| :-----------: | :-----------: | :-----------: | :-----------: | 
| **OpenSource Repo** | <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> |  **Project Page** |  [OpenVoice](https://research.myshell.ai/open-voice) |     
=======
# | | |
# | :-----------: | :-----------: |
# **Join the Community** |   [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |
>>>>>>> acd97c0 (Modify to my useage pattern)

<<<<<<< HEAD
# | | |
# | :-----------: | :-----------: |
# **Join the Community** |   [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |

# </div>
# """
# content = """
# <div>
#   <strong>If the generated voice does not sound like the reference voice, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/QA.md'>this QnA</a>.</strong> <strong>For multi-lingual & cross-lingual examples, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb'>this jupyter notebook</a>.</strong>
#   This online demo mainly supports <strong>English</strong>. The <em>default</em> style also supports <strong>Chinese</strong>. But OpenVoice can adapt to any other language as long as a base speaker is provided.
# </div>
# """
# wrapped_markdown_content = (
#     f"<div style='border: 1px solid #000; padding: 10px;'>{content}</div>"
# )
||||||| parent of acd97c0 (Modify to my useage pattern)
| | |
| :-----------: | :-----------: |
**Join the Community** |   [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |

</div>
"""
content = """
<div>
  <strong>If the generated voice does not sound like the reference voice, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/QA.md'>this QnA</a>.</strong> <strong>For multi-lingual & cross-lingual examples, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb'>this jupyter notebook</a>.</strong>
  This online demo mainly supports <strong>English</strong>. The <em>default</em> style also supports <strong>Chinese</strong>. But OpenVoice can adapt to any other language as long as a base speaker is provided.
</div>
"""
wrapped_markdown_content = f"<div style='border: 1px solid #000; padding: 10px;'>{content}</div>"
=======
# </div>
# """
# content = """
# <div>
#   <strong>If the generated voice does not sound like the reference voice, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/QA.md'>this QnA</a>.</strong> <strong>For multi-lingual & cross-lingual examples, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb'>this jupyter notebook</a>.</strong>
#   This online demo mainly supports <strong>English</strong>. The <em>default</em> style also supports <strong>Chinese</strong>. But OpenVoice can adapt to any other language as long as a base speaker is provided.
# </div>
# """
# wrapped_markdown_content = (
#     f"<div style='border: 1px solid #000; padding: 10px;'>{content}</div>"
# )
>>>>>>> acd97c0 (Modify to my useage pattern)


examples = [
    [
        "今天天气真好，我们一起出去吃饭吧。",
        "default",
        "resources/demo_speaker1.mp3",
        True,
    ],
    [
        "This audio is generated by open voice with a half-performance model.",
        "whispering",
        "resources/demo_speaker2.mp3",
        True,
    ],
    [
        "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
        "sad",
        "resources/demo_speaker0.mp3",
        True,
    ],
]

with gr.Blocks(analytics_enabled=False) as demo:
    # with gr.Row():
    #     with gr.Column():
    #         with gr.Row():
    #             gr.Markdown(
    #                 """
    #                 ## <img src="https://huggingface.co/spaces/myshell-ai/OpenVoice/raw/main/logo.jpg" height="40"/>
    #                 """
    #             )
    #         with gr.Row():
    #             gr.Markdown(markdown_table_v2)
    #         with gr.Row():
    #             gr.Markdown(description)
    #     with gr.Column():
    #         gr.Video(
    #             "https://github.com/myshell-ai/OpenVoice/assets/40556743/3cba936f-82bf-476c-9e52-09f0f417bb2f",
    #             autoplay=True,
    #         )

    # with gr.Row():
    #     gr.HTML(wrapped_markdown_content)

    with gr.Row():
        with gr.Column():
            input_text_gr = gr.Textbox(
                label="Text Prompt",
                info="One or two sentences at a time is better. Up to 200 text characters.",
                value="假如每次想你，都能变成一朵云，那么我将拥有自己的天空。",
            )
            style_gr = gr.Dropdown(
                label="Style",
                info="Select a style of output audio for the synthesised speech. (Chinese only support 'default' now)",
                choices=[
                    "default",
                    "whispering",
                    "cheerful",
                    "terrified",
                    "angry",
                    "sad",
                    "friendly",
                ],
                max_choices=1,
                value="default",
            )
            ref_gr = gr.Audio(
                label="Reference Audio",
                info="Click on the ✎ button to upload your own target speaker audio",
                type="filepath",
<<<<<<< HEAD
                value="resources/dada.wav",
            )
            # tos_gr = gr.Checkbox(
            #     label="Agree",
            #     value=False,
            #     info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
            # )
||||||| parent of acd97c0 (Modify to my useage pattern)
                value="resources/demo_speaker2.mp3",
            )
            tos_gr = gr.Checkbox(
                label="Agree",
                value=False,
                info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
            )
=======
                value="resources/demo_speaker2.mp3",
            )
            # tos_gr = gr.Checkbox(
            #     label="Agree",
            #     value=False,
            #     info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
            # )
>>>>>>> acd97c0 (Modify to my useage pattern)

            tts_button = gr.Button("Send", elem_id="send-btn", visible=True)

        with gr.Column():
            out_text_gr = gr.Text(label="Info")
            audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
            ref_audio_gr = gr.Audio(label="Reference Audio Used")

<<<<<<< HEAD
            # gr.Examples(
            #     examples,
            #     label="Examples",
            #     inputs=[input_text_gr, style_gr, ref_gr],
            #     outputs=[out_text_gr, audio_gr, ref_audio_gr],
            #     fn=predict,
            #     cache_examples=False,
            # )
            tts_button.click(
                predict,
                [input_text_gr, style_gr, ref_gr],  # dada.wav
                outputs=[out_text_gr, audio_gr, ref_audio_gr],
            )
||||||| parent of acd97c0 (Modify to my useage pattern)
            gr.Examples(examples,
                        label="Examples",
                        inputs=[input_text_gr, style_gr, ref_gr, tos_gr],
                        outputs=[out_text_gr, audio_gr, ref_audio_gr],
                        fn=predict,
                        cache_examples=False,)
            tts_button.click(predict, [input_text_gr, style_gr, ref_gr, tos_gr], outputs=[out_text_gr, audio_gr, ref_audio_gr])
=======
            # gr.Examples(
            #     examples,
            #     label="Examples",
            #     inputs=[input_text_gr, style_gr, ref_gr],
            #     outputs=[out_text_gr, audio_gr, ref_audio_gr],
            #     fn=predict,
            #     cache_examples=False,
            # )
            tts_button.click(
                predict,
                [input_text_gr, style_gr, ref_gr],
                outputs=[out_text_gr, audio_gr, ref_audio_gr],
            )
>>>>>>> acd97c0 (Modify to my useage pattern)

demo.queue()
demo.launch(debug=True, show_api=True, share=args.share)
