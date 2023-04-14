import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import json
import math
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import gradio as gr
import numpy as np
from PIL import Image
import numpy as np
import os
from pathlib import Path




LANGUAGES = ['EN','CN','JP']
SPEAKER_ID = 0
COVER = "models/Yuuka/cover.png"
speaker_choice = "Yuuka"
MODEL_ZH_NAME = "早濑优香"
EXAMPLE_TEXT = "先生。今日も全力であなたをアシストしますね。"
#USER_INPUT_TEXT = ""

CONFIG_PATH = "configs/config.json"
MODEL_PATH = "models/Yuuka/Yuuka.pth"

hps = utils.get_hparams_from_file(CONFIG_PATH)
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()

model = net_g.eval()
model = utils.load_checkpoint(MODEL_PATH, net_g, None)

def load_model():
    global hps,net_g,model

    hps = utils.get_hparams_from_file(CONFIG_PATH)
    net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()

    model = net_g.eval()
    model = utils.load_checkpoint(MODEL_PATH, net_g, None)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def tts_fn(text, noise_scale, noise_scale_w, length_scale):
  stn_tst = get_text(text, hps)
  with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([SPEAKER_ID]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
  return  (22050, audio)

def add_model_fn(example_text, cover, speakerID, name_en, name_cn, language):

    # 检查必填字段是否为空
    if not speakerID or not name_en or not language:
        raise gr.Error("Please fill in all required fields!")
        return "Failed to add model"

    ### 保存上传的文件

    # 生成文件路径
    model_save_dir = Path("models")
    model_save_dir = model_save_dir / name_en
    img_save_dir = model_save_dir
    model_save_dir.mkdir(parents=True, exist_ok=True)

    Model_name = name_en + ".pth"
    model_save_dir = model_save_dir / Model_name

    # 保存上传的图片
    if cover is not None:
        img = np.array(cover)
        img = Image.fromarray(img)
        img.save(os.path.join(img_save_dir, 'cover.png'))

    #获取用户输入
    new_model = {
        "name_en": name_en,
        "name_zh": name_cn,
        "cover": img_save_dir / "cover.png",
        "sid": speakerID,
        "example": example_text,
        "language": language,
        "type": "single",
        "model_path": model_save_dir
    }

    #写入json
    with open("models/model_info.json", "r", encoding="utf-8") as f:
        models_info = json.load(f)

    models_info[name_en] = new_model
    with open("models/model_info.json", "w") as f:
        json.dump(models_info, f, cls=CustomEncoder)


    return "Success"

def clear_input_text():
    return ""

def clear_add_model_info():
    return "",None,"","","",""

def get_options():
  with open("models/model_info.json", "r", encoding="utf-8") as f:
    global models_info
    models_info = json.load(f)

  for i,model_info in models_info.items():
    global name_en
    name_en = model_info['name_en']

def reset_options():
  value_model_choice = models_info['Yuuka']['name_en']
  value_speaker_id = models_info['Yuuka']['sid']
  return value_model_choice,value_speaker_id

def refresh_options():
  get_options()
  value_model_choice = models_info[speaker_choice]['name_en']
  value_speaker_id = models_info[speaker_choice]['sid']
  return value_model_choice,value_speaker_id

def change_dropdown(choice):
  global speaker_choice
  speaker_choice = choice
  global COVER
  COVER = str(models_info[speaker_choice]['cover'])
  global MODEL_PATH
  MODEL_PATH = str(models_info[speaker_choice]['model_path'])
  global MODEL_ZH_NAME
  MODEL_ZH_NAME = str(models_info[speaker_choice]['name_zh'])
  global EXAMPLE_TEXT
  EXAMPLE_TEXT = str(models_info[speaker_choice]['example'])

  speaker_id_change = gr.update(value=str(models_info[speaker_choice]['sid']))
  cover_change = gr.update(value='<div align="center">'
                f'<img style="width:auto;height:512px;" src="file/{COVER}">' if COVER else ""
                f'<a><strong>{speaker_choice}</strong></a>'
                                                                                           '</div>')
  title_change = gr.update(value=
                '<div align="center">'
                f'<h3><a><strong>{"语音名称: "}{MODEL_ZH_NAME}</strong></a>'
                f'<h3><strong>{"checkpoint: "}{speaker_choice}</strong>'
                                                                                           '</div>')


  lan_change = gr.update(value=str(models_info[speaker_choice]['language']))

  example_change = gr.update(value=EXAMPLE_TEXT)

  load_model()

  return [speaker_id_change,cover_change,title_change,lan_change,example_change]

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

download_audio_js = """
() =>{{
    let root = document.querySelector("body > gradio-app");
    if (root.shadowRoot != null)
        root = root.shadowRoot;
    let audio = root.querySelector("#tts-audio-{audio_id}").querySelector("audio");
    let text = root.querySelector("#input-text-{audio_id}").querySelector("textarea");
    if (audio == undefined)
        return;
    text = text.value;
    if (text == undefined)
        text = Math.floor(Math.random()*100000000);
    audio = audio.src;
    let oA = document.createElement("a");
    oA.download = text.substr(0, 20)+'.wav';
    oA.href = audio;
    document.body.appendChild(oA);
    oA.click();
    oA.remove();
}}
"""








if __name__ == '__main__':

    with open("models/model_info.json", "r", encoding="utf-8") as f:
        models_info = json.load(f)

    for i, model_info in models_info.items():
        name_en = model_info['name_en']


    theme = gr.themes.Base()

    with gr.Blocks(theme=theme) as interface:
        with gr.Tab("Text to Speech"):
            with gr.Column():
                cover_markdown = gr.Markdown(
                    '<div align="center">'
                    f'<img style="width:auto;height:512px;" src="file/{COVER}">' if COVER else ""
                                                                                               '</div>')
                title_markdown = gr.Markdown(
                    '<div align="center">'
                    f'<h3><a><strong>{"语音名称: "}{MODEL_ZH_NAME}</strong></a>'
                    f'<h3><strong>{"checkpoint: "}{speaker_choice}</strong>'
                    '</div>')

                with gr.Row():
                    with gr.Column(scale=4):
                        input_text = gr.Textbox(
                            label="Input",
                            lines=2,
                            placeholder="Enter the text you want to process here",
                            elem_id=f"input-text-en-{name_en.replace(' ', '')}",
                            scale=2
                        )
                    with gr.Column(scale=1):
                        gen_button = gr.Button("Generate", variant="primary")
                        clear_input_button = gr.Button("Clear")

                with gr.Row():
                    with gr.Column(scale=2):
                        lan = gr.Radio(label="Language", choices=LANGUAGES, value="JP")
                        noise_scale = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="Noise Scale (情感变化程度)",
                                                value=0.6)
                        noise_scale_w = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="Noise Scale w (发音长度)",
                                                  value=0.668)
                        length_scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Length Scale (语速)",
                                                 value=1.0)

                    with gr.Column(scale=1):
                        example_text_box = gr.Textbox(label="Example:",
                                                      value=EXAMPLE_TEXT)

                        output_audio = gr.Audio(label="Output", elem_id=f"tts-audio-en-{name_en.replace(' ', '')}")
                        download_button = gr.Button("Download")

                        # example = gr.Examples(
                        #     examples = [EXAMPLE_TEXT],
                        #     inputs=input_text,
                        #     outputs = output_audio,
                        #     fn=example_tts_fn,
                        #     cache_examples=True
                        # )

            gen_button.click(
                tts_fn,
                inputs=[input_text, noise_scale, noise_scale_w, length_scale],
                outputs=output_audio)
            clear_input_button.click(
                clear_input_text,
                outputs=input_text
            )
            download_button.click(None, [], [], _js=download_audio_js.format(audio_id=f"en-{name_en.replace(' ', '')}"))

        # ------------------------------------------------------------------------------------------------------------------------
        with gr.Tab("AI Singer"):
            input_text_singer = gr.Textbox()

        # ------------------------------------------------------------------------------------------------------------------------
        with gr.Tab("TTS with ChatGPT"):
            input_text_gpt = gr.Textbox()

        # ------------------------------------------------------------------------------------------------------------------------
        with gr.Tab("Settings"):
            with gr.Box():
                gr.Markdown("""# Select Model""")
                with gr.Row():
                    with gr.Column(scale=5):
                        model_choice = gr.Dropdown(label="Model",
                                                   choices=[(model["name_en"]) for name, model in models_info.items()],
                                                   interactive=True,
                                                   value=models_info['Yuuka']['name_en']
                                                   )
                    with gr.Column(scale=5):
                        speaker_id_choice = gr.Dropdown(label="Speaker ID",
                                                        choices=[(str(model["sid"])) for name, model in
                                                                 models_info.items()],
                                                        interactive=True,
                                                        value=str(models_info['Yuuka']['sid'])
                                                        )

                    with gr.Column(scale=1):
                        refresh_button = gr.Button("Refresh", variant="primary")
                        reset_button = gr.Button("Reset")

            model_choice.change(fn=change_dropdown, inputs=model_choice,
                                outputs=[speaker_id_choice, cover_markdown, title_markdown, lan, example_text_box])

            refresh_button.click(fn=refresh_options, outputs=[model_choice, speaker_id_choice])
            reset_button.click(reset_options, outputs=[model_choice, speaker_id_choice])

            with gr.Box():
                gr.Markdown("# Add Model\n"
                            "> *为必填选项\n"
                            "> 添加完成后将**checkpoints**文件放到对应生成的文件夹中"
                            )

                with gr.Row():
                    # file = gr.Files(label = "VITS Model*", file_types=[".pth"])
                    example_text = gr.Textbox(label="Example Text",
                                              lines=16,
                                              placeholder="Enter the example text here", )
                    model_cover = gr.Image(label="Cover")

                    with gr.Column():
                        model_speaker_id = gr.Textbox(label="Speaker List*",
                                                      placeholder="Single speaker model default=0")
                        model_name_en = gr.Textbox(label="name_en*")
                        model_name_cn = gr.Textbox(label="name_cn")
                        model_language = gr.Dropdown(label="Language*",
                                                     choices=LANGUAGES,
                                                     interactive=True)
                        with gr.Row():
                            add_model_button = gr.Button("Add Model", variant="primary")
                            clear_add_model_button = gr.Button("Clear")
                with gr.Box():
                    with gr.Row():
                        message_box = gr.Textbox(label="Message")

            add_model_button.click(add_model_fn,
                                   inputs=[example_text, model_cover, model_speaker_id, model_name_en, model_name_cn,
                                           model_language],
                                   outputs=message_box
                                   )
            clear_add_model_button.click(clear_add_model_info,
                                         outputs=[example_text, model_cover, model_speaker_id, model_name_en,
                                                  model_name_cn, model_language]
                                         )

    interface.queue(concurrency_count=1).launch(debug=True)









