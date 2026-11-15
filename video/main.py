from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
import os
import math, pandas as pd
import hashlib
import requests
from moviepy.video.io.VideoFileClip import VideoFileClip

from IPython.display import Markdown, display
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu

from prompts import get_prompt 

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model, output_loading_info = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto", output_loading_info=True)

directory = './VideoFootage/'
temp_csv = './temp.csv'
prompt_name = "Ablation prompt"


# Inference function

def inference(video, prompt, max_new_tokens=2048, total_pixels=20480 * 32 * 32, min_pixels=64 * 32 * 32, max_frames= 2048, sample_fps = 2):
    """
    Perform multimodal inference on input video and text prompt to generate model response.

    Args:
        video (str or list/tuple): Video input, supports two formats:
            - str: Path or URL to a video file. The function will automatically read and sample frames.
            - list/tuple: Pre-sampled list of video frames (PIL.Image or url).
              In this case, `sample_fps` indicates the frame rate at which these frames were sampled from the original video.
        prompt (str): User text prompt to guide the model's generation.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Default is 2048.
        total_pixels (int, optional): Maximum total pixels for video frame resizing (upper bound). Default is 20480*32*32.
        min_pixels (int, optional): Minimum total pixels for video frame resizing (lower bound). Default is 16*32*32.
        sample_fps (int, optional): ONLY effective when `video` is a list/tuple of frames!
            Specifies the original sampling frame rate (FPS) from which the frame list was extracted.
            Used for temporal alignment or normalization in the model. Default is 2.

    Returns:
        str: Generated text response from the model.

    Notes:
        - When `video` is a string (path/URL), `sample_fps` is ignored and will be overridden by the video reader backend.
        - When `video` is a frame list, `sample_fps` informs the model of the original sampling rate to help understand temporal density.
    """

    messages = [
        {"role": "user", "content": [
                {"video": video,
                "total_pixels": total_pixels,
                "min_pixels": min_pixels,
                "max_frames": max_frames,
                'sample_fps':sample_fps},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True,
                                                                   image_patch_size= 16,
                                                                   return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


for filename in os.listdir(directory):
    if filename.endswith(".mp4"): 
        video = VideoFileClip(os.path.join(directory, filename))
        
        start_time = 0
        part_size = 1 # Minutes
        
        for part, end_time in enumerate(range(part_size*60, int(video.duration), part_size*60)):
            split_file = directory +'split/'+ filename + '_' + str(part)+'.mp4'
           
            if not os.path.isfile(split_file):
                clip = video.subclipped(start_time, end_time)
                clip.write_videofile(split_file)

            prompt = get_prompt(prompt_name)
            response = inference(split_file, prompt)

            if os.path.isfile(temp_csv):
                os.remove(temp_csv)
          
            with open(temp_csv, "a") as f:
                f.write(response)

            df = pd.read_csv(temp_csv, header=None)

            try:
                df['start'] = df[0] + pd.Timedelta(minutes=part*part_size)
            except:
                pass
                
            df['video'] = filename[:15]
            df['prompt'] = prompt_name
            df['part'] = part
        
            df.to_csv('./csv/'+filename[:15]+'_'+ prompt_name +'.csv', mode='a', index=False, header=False)
            
            if os.path.isfile(temp_csv):
                os.remove(temp_csv)

