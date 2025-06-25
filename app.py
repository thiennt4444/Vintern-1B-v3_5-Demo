import spaces
import argparse
from ast import parse
import datetime
import json
import os
import time
import hashlib
import re
import torch
import gradio as gr
import requests
import random
from filelock import FileLock
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from models import load_image
from constants import LOGDIR, DEFAULT_IMAGE_TOKEN
from utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
    load_image_from_base64,
    get_log_filename,
)
from threading import Thread
import traceback
# import torch
from conversation import Conversation
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
import subprocess

subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

torch.set_default_device('cuda')

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "Vintern-1B-3.5-Demo Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)


@spaces.GPU(duration=10)
def make_zerogpu_happy():
    pass


def write2file(path, content):
    lock = FileLock(f"{path}.lock")
    with lock:
        with open(path, "a") as fout:
            fout.write(content)


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def init_state(state=None):
    if state is not None:
        del state
    return Conversation()

def vote_last_response(state, liked, request: gr.Request):
    conv_data = {
        "tstamp": round(time.time(), 4),
        "like": liked,
        "model": 'Vintern-3B-beta',
        "state": state.dict(),
        "ip": request.client.host,
    }
    write2file(get_log_filename(), json.dumps(conv_data) + "\n")


def upvote_last_response(state, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, True, request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3


def downvote_last_response(state, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, False, request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3


def vote_selected_response(
    state, request: gr.Request, data: gr.LikeData
):
    logger.info(
        f"Vote: {data.liked}, index: {data.index}, value: {data.value} , ip: {request.client.host}"
    )
    conv_data = {
        "tstamp": round(time.time(), 4),
        "like": data.liked,
        "index": data.index,
        "model": 'Vintern-3B-beta',
        "state": state.dict(),
        "ip": request.client.host,
    }
    write2file(get_log_filename(), json.dumps(conv_data) + "\n")
    return


def flag_last_response(state, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    # state.messages[-1][-1] = None
    state.update_message(Conversation.ASSISTANT, content='', image=None, idx=-1)
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = init_state()
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 5


def add_text(state, message, system_prompt, request: gr.Request):
    if not state:
        state = init_state()
    images = message.get("files", [])
    text = message.get("text", "").strip()
    # logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    # import pdb; pdb.set_trace()
    textbox = gr.MultimodalTextbox(value=None, interactive=False)
    if len(text) <= 0 and len(images) == 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), textbox) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            textbox = gr.MultimodalTextbox(
                value={"text": moderation_msg}, interactive=True
            )
            return (state, state.to_gradio_chatbot(), textbox) + (no_change_btn,) * 5
    images = [Image.open(path).convert("RGB") for path in images]

    # Init again if send the second image
    if len(images) > 0 and len(state.get_images(source=state.USER)) > 0:
        state = init_state(state)

    # Upload the first image
    if len(images) > 0 and len(state.get_images(source=state.USER)) == 0:
        if len(state.messages) == 0: ## In case the first message is an image
            text = DEFAULT_IMAGE_TOKEN + "\n" + system_prompt + "\n" + text
        else: ## In case the image is uploaded after some text messages
           first_user_message = state.messages[0]['content']
           state.update_message(Conversation.USER, DEFAULT_IMAGE_TOKEN + "\n" + first_user_message, None, 0)

    # If the first message is text
    if len(images) == 0 and len(state.get_images(source=state.USER)) == 0 and len(state.messages) == 0:
        text = system_prompt + "\n" + text


    state.set_system_message(system_prompt)
    state.append_message(Conversation.USER, text, images)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), textbox) + (
        disable_btn,
    ) * 5

model_name = "5CD-AI/Vintern-3B-beta"
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

@spaces.GPU
def predict(state,
            image_path,
            max_input_tiles=6, 
            temperature=1.0,
            max_output_tokens=700,
            top_p=0.7,
            repetition_penalty=2.5,
            do_sample=False):
        
        # history = state.get_prompt()[:-1]
        # logger.info(f"==== History ====\n{history}")

        logger.info(f"function predict")

        generation_config = dict(temperature=temperature,
                                 max_new_tokens=max_output_tokens, 
                                 top_p=top_p, 
                                 do_sample=do_sample, 
                                 num_beams = 3, 
                                 repetition_penalty=repetition_penalty)

        pixel_values = None
        if image_path is not None:
            pixel_values = load_image(image_path, max_num=max_input_tiles).to(torch.bfloat16).cuda()
            
            if pixel_values is not None:
                logger.info(f"==== Lenght Pixel values ====\n{len(pixel_values)}")

               # Check the first user message to see if it is an image
                index, first_user_message = state.get_user_message(source=state.USER, position='first')
                if first_user_message is not None and \
                    DEFAULT_IMAGE_TOKEN not in first_user_message:
                    state.update_message(state.USER, DEFAULT_IMAGE_TOKEN + "\n" + first_user_message, None, index)

        history = state.get_history()
        logger.info(f"====  History ====\n{history}")
        _, message = state.get_user_message(source=state.USER, position='last')

        
        response, conv_history = model.chat(tokenizer, 
                                            pixel_values, 
                                            message, 
                                            generation_config, 
                                            history=history,
                                            return_history=True)
        logger.info(f"==== Conv History ====\n{conv_history}")
        return response, conv_history

def ai_bot(
    state,
    temperature,
    do_sample,
    top_p,
    repetition_penalty,
    max_new_tokens,
    max_input_tiles,
    request: gr.Request,
):
    logger.info(f"function ai_bot")
    logger.info(f"ai_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    if hasattr(state, "skip_next") and state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=False),
        ) + (no_change_btn,) * 5
        return

    if model is None:
        state.update_message(Conversation.ASSISTANT, server_error_msg)
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=False),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    all_images = state.get_images(source=state.USER)
    all_image_paths = [state.save_image(image) for image in all_images]

    state.append_message(Conversation.ASSISTANT, state.streaming_placeholder)
    yield (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(interactive=False),
    ) + (disable_btn,) * 5

    try:
        # Stream output
        logger.info(f"==== Image paths ===={all_image_paths}")

        response, _ = predict(state,
                            all_image_paths[0] if len(all_image_paths) > 0 else None,
                            max_input_tiles, 
                            temperature, 
                            max_new_tokens,
                            top_p, 
                            repetition_penalty,
                            do_sample)

        # response = "This is a test response"
        buffer = ""
        for new_text in response:
            buffer += new_text
                
            state.update_message(Conversation.ASSISTANT, buffer + state.streaming_placeholder, None)
            yield (
                state,
                state.to_gradio_chatbot(),
                gr.MultimodalTextbox(interactive=False),
            ) + (disable_btn,) * 5

    except Exception as e:
        logger.error(f"Error in ai_bot: {e} \n{traceback.format_exc()}")
        state.update_message(Conversation.ASSISTANT, server_error_msg, None)
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=True),
        ) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    ai_response = state.return_last_message()

    logger.info(f"==== AI response ===={ai_response}")
   
    state.end_of_current_turn()

    finish_tstamp = time.time()
    elapsed_time = round(finish_tstamp - start_tstamp, 4)
    logger.info(f"ai_bot processing time: {elapsed_time} seconds")

    yield (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(interactive=True),
    ) + (enable_btn,) * 5

    logger.info(f"{buffer}")
    data = {
        "tstamp": round(finish_tstamp, 4),
        "like": None,
        "model": model_name,
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "elapsed_time": elapsed_time,
        "state": state.dict(),
        "images": all_image_paths,
        "ip": request.client.host,
    }
    write2file(get_log_filename(), json.dumps(data) + "\n")


def ocr_bot(
    state,
    temperature,
    do_sample,
    top_p,
    repetition_penalty,
    max_new_tokens,
    max_input_tiles,
    request: gr.Request,
):
    logger.info(f"function ocr_bot")
    logger.info(f"ocr_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    if hasattr(state, "skip_next") and state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=False),
        ) + (no_change_btn,) * 5
        return

    if model is None:
        state.update_message(Conversation.ASSISTANT, server_error_msg)
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=False),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    all_images = state.get_images(source=state.USER)
    all_image_paths = [state.save_image(image) for image in all_images]

    state.append_message(Conversation.ASSISTANT, state.streaming_placeholder)
    yield (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(interactive=False),
    ) + (disable_btn,) * 5

    try:

        logger.info(f"==== Image paths ====\n{all_image_paths}")
        
        response, _ = predict(state,
                            all_image_paths[0] if len(all_image_paths) > 0 else None,
                            max_input_tiles, 
                            temperature, 
                            max_new_tokens,
                            top_p, 
                            repetition_penalty,
                            do_sample=False)  # Tắt sampling để có kết quả nhất quán

        buffer = response
        # Trả về nội dung OCR cuối cùng
        state.update_message(Conversation.ASSISTANT, buffer, None)
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(value=buffer, interactive=True),
        ) + (enable_btn,) * 5

    except Exception as e:
        logger.error(f"Error in ocr_bot: {e} \n{traceback.format_exc()}")
        state.update_message(Conversation.ASSISTANT, server_error_msg, None)
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=True),
        ) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        ) 
        return

    ai_response = state.return_last_message()
    state.end_of_current_turn()
  
    

# <h1 style="font-size: 28px; font-weight: bold;">Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling</h1>
title_html = """
<div style="text-align: center;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/6336b5c831efcb5647f00170/-G297bBqMzYvTbD6_Bkd9.png" style="height: 95px; width: 100%;">
    <p style="font-size: 20px;">❄️Vintern-3B-beta❄️</p>
    <p style="font-size: 14px;">An Efficient Multimodal Large Language Model for Vietnamese🇻🇳</p>
    <a href="https://huggingface.co/papers/2408.12480" style="font-size: 13px;">[📖 Vintern Paper]</a>
    <a href="https://huggingface.co/5CD-AI" style="font-size: 13px;">[🤗 Huggingface]</a>
</div>
"""

description_html = """
<div style="text-align: left;">
    <p style="font-size: 12px;">Vintern-1B-v3.5 is the latest in the Vintern series, bringing major improvements over v2 across all benchmarks. This continuous fine-tuning Version enhances Vietnamese capabilities while retaining strong English performance. It excels in OCR, text recognition, and Vietnam-specific document understanding.</p>
</div>
"""

tos_markdown = """
### Terms of use
By using this service, users are required to agree to the following terms:
It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""


# .gradio-container {margin: 5px 10px 0 10px !important};
block_css = """
.gradio-container {margin: 0.1% 1% 0 1% !important; max-width: 98% !important;};
#buttons button {
    min-width: min(120px,100%);
}
.gradient-text {
    font-size: 28px;
    width: auto;
    font-weight: bold;
    background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
}
.plain-text {
    font-size: 22px;
    width: auto;
    font-weight: bold;
}
"""

js = """
function createWaveAnimation() {
    const text = document.getElementById('text');
    var i = 0;
    setInterval(function() {
        const colors = [
            'red, orange, yellow, green, blue, indigo, violet, purple',
            'orange, yellow, green, blue, indigo, violet, purple, red',
            'yellow, green, blue, indigo, violet, purple, red, orange',
            'green, blue, indigo, violet, purple, red, orange, yellow',
            'blue, indigo, violet, purple, red, orange, yellow, green',
            'indigo, violet, purple, red, orange, yellow, green, blue',
            'violet, purple, red, orange, yellow, green, blue, indigo',
            'purple, red, orange, yellow, green, blue, indigo, violet',
        ];
        const angle = 45;
        const colorIndex = i % colors.length;
        text.style.background = `linear-gradient(${angle}deg, ${colors[colorIndex]})`;
        text.style.webkitBackgroundClip = 'text';
        text.style.backgroundClip = 'text';
        text.style.color = 'transparent';
        text.style.fontSize = '28px';
        text.style.width = 'auto';
        text.textContent = 'Vintern-1B';
        text.style.fontWeight = 'bold';
        i += 1;
    }, 200);
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    // console.log(url_params);
    // console.log('hello world...');
    // console.log(window.location.search);
    // console.log('hello world...');
    // alert(window.location.search)
    // alert(url_params);
    return url_params;
}
"""


def build_demo():
    textbox = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image", "video"],
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    with gr.Blocks(
        title="❄️ Vintern-3B-beta-Demo ❄️",
        theme="NoCrypt/miku",
        css=block_css,
        js=js,
    ) as demo:
        state = gr.State()

        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(title_html)

                with gr.Accordion("Settings", open=False) as setting_row:
                    system_prompt = gr.Textbox(
                        value="Bạn là một trợ lý AI đa phương thức hữu ích, hãy trả lời câu hỏi người dùng một cách chi tiết.",
                        label="System Prompt",
                        interactive=True,
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    do_sample = gr.Checkbox(
                        label="Sampling",
                        value=False,
                        interactive=True,
                    )

                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.2,
                        step=0.02,
                        interactive=True,
                        label="Repetition penalty",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=700,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )
                    max_input_tiles = gr.Slider(
                        minimum=1,
                        maximum=12,
                        value=6,
                        step=1,
                        interactive=True,
                        label="Max input tiles (control the image size)",
                    )
                examples = gr.Examples(
                    examples=[
                        [
                            {
                                "files": [
                                    "samples/1.jpg",
                                ],
                                "text": "Hãy trích xuất thông tin từ hình ảnh này và trả về kết quả dạng markdown.",
                            }
                        ],
                        [
                            {
                                "files": [
                                    "samples/2.png",
                                ],
                                "text": "Bạn là một nhà sáng tạo nội dung tài năng. Hãy viết một kịch bản quảng cáo cho sản phẩm này.",
                            }
                        ],
                        [
                            {
                                "files": [
                                    "samples/3.jpeg",
                                ],
                                "text": "Hãy viết lại một email cho các chủ hộ về nội dung của bảng thông báo.",
                            }
                        ],
                        [
                            {
                                "files": [
                                    "samples/6.jpeg",
                                ],
                                "text": "Hãy viết trích xuất nội dung của hoá đơn dạng JSON.",
                            }
                        ],
                    ],
                    inputs=[textbox],
                )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Vintern-3B-beta-Demo",
                    height=580,
                    show_copy_button=True,
                    show_share_button=True,
                    avatar_images=[
                        "assets/human.png",
                        "assets/assistant.png",
                    ],
                    bubble_full_width=False,
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(
                        value="🔄  Regenerate", interactive=False
                    )
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
                    ocr_btn = gr.Button(value="📝  OCR", interactive=True, variant="secondary")
                with gr.Row():
                    gr.HTML(description_html)

        gr.Markdown(tos_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        downvote_btn.click(
            downvote_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        chatbot.like(
            vote_selected_response,
            [state],
            [],
        )
        flag_btn.click(
            flag_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        regenerate_btn.click(
            regenerate,
            [state, system_prompt],
            [state, chatbot, textbox] + btn_list,
        ).then(
            ai_bot,
            [
                state,
                temperature,
                do_sample,
                top_p,
                repetition_penalty,
                max_output_tokens,
                max_input_tiles,
            ],
            [state, chatbot, textbox] + btn_list,
        )
        clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

        textbox.submit(
            add_text,
            [state, textbox, system_prompt],
            [state, chatbot, textbox] + btn_list,
        ).then(
            ai_bot,
            [
                state,
                temperature,
                do_sample,
                top_p,
                repetition_penalty,
                max_output_tokens,
                max_input_tiles,
            ],
            [state, chatbot, textbox] + btn_list,
        )
        submit_btn.click(
            add_text,
            [state, textbox, system_prompt],
            [state, chatbot, textbox] + btn_list,
        ).then(
            ai_bot,
            [
                state,
                temperature,
                do_sample,
                top_p,
                repetition_penalty,
                max_output_tokens,
                max_input_tiles,
            ],
            [state, chatbot, textbox] + btn_list,
        )
        
        # OCR button handler
        ocr_btn.click(
            add_text,
            [state, textbox, system_prompt],
            [state, chatbot, textbox] + btn_list,
        ).then(
            ocr_bot,
            [
                state,
                temperature,
                do_sample,
                top_p,
                repetition_penalty,
                max_output_tokens,
                max_input_tiles,
            ],
            [state, chatbot, textbox] + btn_list,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    logger.info(args)
    demo = build_demo()
    demo.queue(api_open=False).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=args.concurrency_count,
    )
