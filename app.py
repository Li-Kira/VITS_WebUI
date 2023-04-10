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
speaker_id = 0
config_path = "configs/config.json"
model_path = "G:\AI\Model\VITS\Yuuka\G_4000.pth"
cover = "models/Yuuka/cover.png"

hps = utils.get_hparams_from_file(config_path)
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()

model = net_g.eval()
model = utils.load_checkpoint(model_path, net_g, None)



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
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
  return  (22050, audio)

def add_model_fn(example_text, cover, SpeakerID, name_en, name_cn, language):



    # 检查必填字段是否为空
    if not speaker_id or not name_en or not language:
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
        img.save(os.path.join(img_save_dir, 'cover_white_background.png'))



    #获取用户输入
    new_model = {
        "name_en": name_en,
        "name_zh": name_cn,
        "cover": img_save_dir / "cover.png",
        "sid": SpeakerID,
        "example": "それに新しいお菓子屋さんも出来てみんな買いものを楽しんでいます！",
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

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def clear_input_text():
    return ""


def clear_add_model_info():
    return "",None,"","","",""





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


    ### GUI
    theme = gr.themes.Base()
    with gr.Blocks(theme=theme) as interface:
        with gr.Tab("Text to Speech"):
            with gr.Column():
                gr.Markdown(
                    '<div align="center">'
                    f'<img style="width:auto;height:512px;" src="file/{cover}">' if cover else ""
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
                        lan = [gr.Radio(label="Language", choices=LANGUAGES, value="JP")]
                        noise_scale = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="Noise Scale (情感变化程度)",
                                                value=0.6)
                        noise_scale_w = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="Noise Scale w (发音长度)",
                                                  value=0.668)
                        length_scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Length Scale (语速)",
                                                 value=1.0)

                    with gr.Column(scale=1):
                        output_audio = gr.Audio(label="Output", elem_id=f"tts-audio-en-{name_en.replace(' ', '')}")
                        download_button = gr.Button("Download")

            # clear_input_button.click()
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
                                                   value=models_info['yuuka']['name_en']
                                                   )
                    with gr.Column(scale=5):
                        speaker_id = gr.Dropdown(label="Speaker ID",
                                                 choices=[(str(model["sid"])) for name, model in models_info.items()],
                                                 interactive=True,
                                                 value=str(models_info['yuuka']['sid'])
                                                 )

                    with gr.Column(scale=1):
                        refresh_button = gr.Button("Refresh", variant="primary")
                        reset_button = gr.Button("Reset")

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














