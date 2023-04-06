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
import tempfile



LANGUAGES = ['EN','CN','JP']
speaker_id = 0



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



config_path = "configs/config.json"
model_path = "G:\AI\Model\VITS\Yuuka\G_4000.pth"

hps = utils.get_hparams_from_file(config_path)
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()

model = net_g.eval()
model = utils.load_checkpoint(model_path, net_g, None)






with gr.Blocks() as interface:
  with gr.Tab("Text to Speech"):
    with gr.Row():
      input_text = gr.Textbox(
          label="Input",
          lines=5,
          placeholder="Enter the text you want to process here")
      gen_button = gr.Button("Generate")
    with gr.Row():
      with gr.Column():
        lan = [gr.Radio(label="Language", choices=LANGUAGES, value="JP")]
        noise_scale = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label = "Noise Scale (情感变化程度)", value = 0.6)
        noise_scale_w = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label = "Noise Scale w (发音长度)", value = 0.668)
        length_scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label = "Length Scale (语速)", value=1.0)

      with gr.Column():
        output_audio = gr.Audio(label="Output")
        download_button = gr.Button("Download")

    gen_button.click(
        tts_fn,
        inputs = [input_text, noise_scale, noise_scale_w, length_scale],
        outputs = output_audio
        )
    #download_button.click(None, [], [], _js=download_audio_js.format(audio_id=f"en-{name_en.replace(' ', '')}"))


  with gr.Tab("TTS with ChatGPT"):
    input_text_gpt = gr.Textbox()


  with gr.Tab("Settings"):
    model_name = gr.Dropdown(label = "model")





if __name__ == '__main__':
    interface.launch()







