# VITS_WebUI

> A WebUI for text-to-speech inference based on the Vits model
>
> 一个基于Vits模型的文本转语音推理WebUI


**Preview 项目预览：**

![image](https://user-images.githubusercontent.com/62274988/231185600-feb2fec3-24cd-4695-beab-0362ef4286f6.png)

![image](https://user-images.githubusercontent.com/62274988/231162669-10c84a50-1dd9-45f3-a4c9-85cd9daaf0d5.png)



## Features

- Text to Speech inference 文本转语音推理
- Generate parameter adjustment 生成参数调整
- Export generated audio 导出生成音频
- Import model information 导入模型信息：name_en、name_ch、Speaker ID、Example text、Cover、Language
- Import model error message 导入模型错误提示
- Display model information 模型信息显示
- Switch checkpoints 切换模型
- Display example text 实例文本显示

## Terms of Use

🚨**Please resolve the dataset authorization issues by yourself. Any problems caused by using unauthorized datasets for training are at your own risk and responsibility.This project does not provide datasets or model files.**

🚨**请自行解决数据集授权问题，任何因使用未经授权的数据集进行训练而造成的问题，由您自行承担风险和责任。本项目不提供数据集以及模型文件。**


## Getting started

>### using Gradio 3.25.0

 1.Install requirment 安装依赖

```
pip install - r requirements.txt
```

2.Run`setup.py` 启动环境

```
%cd VITS_WebUI\monotonic_align
!python setup.py build_ext --inplace
```

> 单调对齐搜索，一种搜索最大化的，由归一化流参数化的数据的对齐的方法。

3.Run interface 启动界面

```
%cd VITS_WebUI
!python app.py
```

或者你也可以执行此文件`GUI_Usage.ipynb`

4.View in browser 在浏览器中查看

```
http://127.0.0.1:7860/
```

5.在**Settings**中导入模型信息，并将训练好的checkpoints文件改名为对应`name_en.pth`并放到**models**目录下对应生成的文件夹中。
> 首次运行请在`models/Yuuka`中放入任意checkpoints


## Detail 

> 如果你想在此项目上修改，可以参考这里



### Inference-related functions

- `get_text(text, hps)`：文本预处理函数，能够将**输入文本**转换为**音素**
- `tts_fn(text, noise_scale, noise_scale_w, length_scale)`：推理函数，给定三个超参数**noise_scale**、**noise_scale_w**、**length_scale**用于控制生成音频的**情感变化长度**、**发音长度**、**语速**



### Gradio-related function

> **Gradio具有丰富的案例Demo和详尽的开发文档，入门请参考这些网站**
>
> - Gradio案例：https://gradio.app/demos/
> - Gradio文档：https://gradio.app/docs/

#### 

- `get_options()`：从`models/model_info.json`该json文件中获取模型信息

- `add_model_fn()`：导入模型信息函数，支持导入以下参数：**example_text**，**cover**，**speakerID**，**name_en**，**name_cn**，**language**

  > **speakerID**、**name_en**、**language**为必填选项，如果为空则会提升错误
  >
  > ```python
  > # 检查必填字段是否为空
  > if not SPEAKER_ID or not name_en or not language:
  >     raise gr.Error("Please fill in all required fields!")
  >     return "Failed to add model"
  > ```
  >
  > 如果要执行`gr.Error`，必须在**click**方法中添加一个组件到**outputs**，否则会报错，这里我选择的是一个**Textbox组件**

- `change_dropdown()`：更新模型信息函数，当**Dropdown**选取新的模型时，实时地更新

  > **注意：**
  >
  > - 由于Gradio的`update()`方法无法更新**Example组件**，于是用Textbox来替代示例
  >
  > - 声明了**Global**全局变量
  >
  > - 需要在`Blocks`里面添加**change**函数
  >
  >   ```python
  >   model_choice.change(fn=change_dropdown, inputs=model_choice, outputs=[speaker_id_choice,cover_markdown,title_markdown,lan,example_text_box])
  >   ```

- `CustomEncoder(json.JSONEncoder)`：json编码器类，用于解决在执行`json.dump()`时出现的格式错误

- `download_audio_js`：JS方法，用来下载生成的音频，保存的文件名为**Input Textbox**中的文本截取

  > 需要为作为**Inputs**的**Textbox组件**添加一个**elem_id**
  >
  > ```python
  > elem_id=f"input-text-en-{name_en.replace(' ','')}",
  > ```
  >
  > 以及为作为**Outputs**的**Audio组件**添加一个elem_id
  >
  > ```python
  > output_audio = gr.Audio(label="Output", elem_id=f"tts-audio-en-{name_en.replace(' ','')}")
  > ```
  >
  > 该方法在**download_button**被点击时执行
  >
  > ```python
  > download_button.click(None, [], [], _js=download_audio_js.format(audio_id=f"en-{name_en.replace(' ', '')}"))
  > ```



### Layout

- **Blocks:**

  > **能创建自定义的UI，搭配CSS食用更佳：** https://gradio.app/custom-CSS-and-JS/

- **Markdown:**

  `gr.Markdown`支持Markdown格式的输入，你也可以输入html文本

  > ```python
  > gr.Markdown("""# Select Model""")
  > ```

  > ```python
  > cover_markdown = gr.Markdown(
  >     '<div align="center">'
  >     f'<img style="width:auto;height:512px;" src="file/{COVER}">' if COVER else ""
  >                                                                                '</div>')
  > title_markdown = gr.Markdown(
  >     '<div align="center">'
  >     f'<h3><a><strong>{"语音名称: "}{MODEL_ZH_NAME}</strong></a>'
  >     f'<h3><strong>{"checkpoint: "}{speaker_choice}</strong>'
  >                                                                                '</div>')
  > ```

- **Columns and Nesting**

  你可以使用`gr.Box()`、`gr.Row()`、`gr.Column`来控制组件的布局

  > 只有**Column**能够使用**scale**来控制大小，如果想控制**Row**的大小，你可以采取以下嵌套方式
  >
  > ```python
  > with gr.Box():
  >     with gr.Row():
  >         with gr.Column(scale = 2):
  >             #Components 
  >         with gr.Column(scale = 1):
  >             #Components
  > ```

- **Theme：**

  我这里使用的是**Base**主题，你可以查看官方文档选择你想要的主题：https://gradio.app/theming-guide/

  > ```python
  > theme = gr.themes.Base()
  > 
  > with gr.Blocks(theme=theme) as interface:
  >     #Components 
  >     
  > interface.queue(concurrency_count=1).launch(debug=True)
  > ```
  >
  > `queue(concurrency_count=1)`开启可以显示进程的执行时间，`debug`这个参数默认是**False**，开启后可以使用debug功能，定位到错误信息。







## Reference

- 论文地址:https://arxiv.org/abs/2106.06103
- 论文代码：https://github.com/jaywalnut310/vits
- Vits微调：https://github.com/SayaSS/vits-finetuning
