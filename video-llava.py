import av
import numpy as np
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, BitsAndBytesConfig
from ipex_llm import optimize_model
from ipex_llm.optimize import low_memory_init, load_low_bit
from pathlib import Path
import time
from PIL import Image
import requests
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

#from intel_extension_for_transformers.transformers import RtnConfig

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
transform = Compose(
    [
        # UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ShortSideScale(size=224),
        CenterCropVideo(224),
        RandomHorizontalFlipVideo(p=0.5),
    ]
)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def load_and_transform_video(
        video_path,
        transform,
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8):
    
    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

    video_data = []
    for frame_idx in frame_id_list:
        cv2_vr.set(1, frame_idx)
        _, frame = cv2_vr.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
    cv2_vr.release()
    video_data = torch.stack(video_data, dim=1)
    video_outputs = transform(video_data)
    #video_outputs = video_data
    return video_outputs

def load_video(
        video_path,
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8):
    
    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
    frames=[]

    video_data = []
    for frame_idx in frame_id_list:
        cv2_vr.set(1, frame_idx)
        _, frame = cv2_vr.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cv2_vr.release()
    return np.stack(frames)


# Initialize model and processor
model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", trust_remote_code=True) #, torch_dtype=torch.float16)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", trust_remote_code=True)

#print(f"vision: model.video_tower: {model.video_tower}")

optimzed_llm_saved_path = "./video_llava_llm_sym_int4"
#woq_quantization_config = RtnConfig(compute_dtype="fp16", weight_dtype="int4_sym", scale_dtype="fp16", group_size=64)

model = optimize_model(model, low_bit="sym_int4").to('xpu')

#optimized_vp_model.save_low_bit("./video_llava_procesor_sym_int4")
#print("save optimized llm")

########################
# Warm-up (1 iteration)
########################
for i in range(1):
    prompt = "USER: <video>Why is this video funny? ASSISTANT:"
    video_path = "sample_demo_13.mp4"
    container = av.open(video_path)

    # Sample uniformly 8 frames from the video
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    print(f"total_frames: {total_frames}, indices: {indices}")
    clip = read_video_pyav(container, indices)

    start_time = time.time();
    print(f"in start: {start_time}")
    inputs = processor(text=prompt, videos=clip, return_tensors="pt")

    # Move inputs to the XPU
    inputs = {k: v.to('xpu') for k, v in inputs.items()}

    end_time = time.time()
    print(f"in end: {end_time}")

    start_time = time.time()
    print(f"gen start: {start_time}")
    # Generate response with memory considerations
    try:
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_length=80)
            result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(result)
    except RuntimeError as e:
        print("Caught a runtime error during generation:", e)
    except Exception as e:
        print("Caught an unknown exception during generation:", e)

    end_time=time.time()
    print(f"gen elapsed time: {end_time-start_time} s")
    
#################################################################
# gradio example
#################################################################    
import gradio as gr
#from videollava.conversation import conv_templates, SeparatorStyle
import ffmpeg
import cv2

examples_dir = Path("./examples")

def save_video_as_mp4(in_file_name, out_file_name):
    cap=cv2.VideoCapture(in_file_name)
    while cap.isOpened():
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        fps=15
        if int(major_ver)  < 3 :
           fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
           print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
           fps = cap.get(cv2.CAP_PROP_FPS)
           print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        ret, frame = cap.read()
        if ret:
            i_width,i_height = frame.shape[1],frame.shape[0]
            break

    print(f"width: {i_width}, height: {i_height}")
    cap.release()
    
    process=None

    if i_height > 720:
        process = (
        ffmpeg
            #.input(in_file_name.decode('string_escape'),format="webm",s='{}x{}'.format(i_width,i_height))
            .input(Path(in_file_name).as_posix())
            .filter("scale",-1, 720)
            .output(out_file_name,pix_fmt='yuv420p',vcodec='libx264',r=fps,crf=37)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
    else:
        process = (
        ffmpeg
            #.input(in_file_name.decode('string_escape'),format="webm",s='{}x{}'.format(i_width,i_height))
            .input(Path(in_file_name).as_posix())
            .output(out_file_name,pix_fmt='yuv420p',vcodec='libx264',r=fps,crf=37)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )       
    out, err = process.communicate()
    print(f"stdout: {out}  stderr: {err}")

    return True

def generate(image, video, textbox_in):   
    prompt_text = ""

    if video is not None:
       prompt_text = prompt_text + "<video>\n"
    if image is not None:
       prompt_text = "<image>\n" + prompt_text
       
    prompt_text = f"USER: {prompt_text} {textbox_in} ASSISTANT:"

    video_clip = None
    image_clip = None
    
    print(f"prompt: {prompt_text}")

    if video is not None:
        video=Path(video).as_posix()
        print(f"Video: {video}")
        extension=video.rsplit(".",1)[-1]
        print(f"Ext: {extension}")
        if extension=="webm":
           out_video_fn="out_video.mp4"
           save_video_as_mp4(video, out_video_fn)
           video=out_video_fn
        
        video_clip=load_video(video)
        #container = av.open(video)
        # Sample uniformly 8 frames from the video
        #total_frames = container.streams.video[0].frames
        #indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        #video_clip = read_video_pyav(container, indices)
        
    if image is not None:
        image=Path(image).as_posix()
        print(f"Image: {image}")
        image_clip = Image.open(image)
 
    inputs = processor(text=prompt_text, images=image_clip, videos=video_clip, padding=True, return_tensors="pt")
 

    # Move inputs to the XPU
    inputs = {k: v.to('xpu') for k, v in inputs.items()}
    error_output = ""
    result = ""
    start_time = time.time()
    print(f"gen start: {start_time}")
    # Generate response with memory considerations
    try:
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_length=80)
            result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            #return result
            #print(result)
    except RuntimeError as e:
        print("Caught a runtime error during generation:", e)
        error_output = e
    except Exception as e:
        print("Caught an unknown exception during generation:", e)
        error_output = e

    end_time=time.time()
    print(f"gen elapsed time: {end_time-start_time} s")
    
    if error_output != "":
       return f"<error> {error_output}"
    else:
       return result
        
        
"""
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    generate_kwargs = dict(
        input_ids=input_ids,
        images=images_tensor,
        max_new_tokens=1024,
        temperature=0.2,
        do_sample=True,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )

    output_ids = ov_model.generate(**generate_kwargs)

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    return outputs
"""

demo = gr.Interface(
    generate,
    [
        gr.Image(label="Input Image", type="filepath"),
        gr.Video(label="Input Video"),
        gr.Textbox(label="Question"),
    ],
    gr.Textbox(lines=10),
    examples=[
        [
            f"{examples_dir}/extreme_ironing.jpg",
            None,
            "What is unusual about this image?",
        ],
        [
            f"{examples_dir}/waterview.jpg",
            None,
            "What are the things I should be cautious about when I visit here?",
        ],
        [
            f"{examples_dir}/desert.jpg",
            None,
            "If there are factual errors in the questions, point it out; if not, proceed answering the question. What’s happening in the desert?",
        ],
        [
            None,
            f"{examples_dir}/sample_demo_1.mp4",
            "Why is this video funny?",
        ],
        [
            None,
            f"{examples_dir}/sample_demo_3.mp4",
            "Can you identify any safety hazards in this video?",
        ],
        [
            None,
            f"{examples_dir}/sample_demo_9.mp4",
            "Describe the video.",
        ],
        [
            None,
            f"{examples_dir}/sample_demo_22.mp4",
            "Describe the activity in the video.",
        ],
        [
            f"{examples_dir}/sample_img_22.png",
            f"{examples_dir}/sample_demo_22.mp4",
            "Are the instruments in the pictures used in the video?",
        ],
        [
            f"{examples_dir}/sample_img_13.png",
            f"{examples_dir}/sample_demo_13.mp4",
            "Does the flag in the image appear in the video?",
        ],
        [
            f"{examples_dir}/sample_img_8.png",
            f"{examples_dir}/sample_demo_8.mp4",
            "Are the image and the video depicting the same place?",
        ],
        [
            None,
            f"{examples_dir}/rag_videos/op_5_0320241915.mp4",
            "Describe the video.",
        ],
    ],
    title="Video-LLaVA🚀",
    allow_flagging="never",
)
try:
    demo.queue().launch(debug=True)
except Exception:
    demo.queue().launch(share=True, debug=True)
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/
