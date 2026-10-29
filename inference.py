import argparse
import os
import warnings
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
import cv2
import numpy as np
from typing import List
from einops import rearrange
import librosa
import soundfile as sf
import torchaudio
import imageio
import subprocess
import shutil
import csv
import time
from datetime import datetime
from pathlib import Path
from denoiser import pretrained
from denoiser.dsp import convert_audio

from src.pipelines.face_animate import FaceAnimatePipeline
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from image_processor import ImageProcessor
from src.utils.util import save_video, filter_non_none


warnings.filterwarnings("ignore")

class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    
    Args:
        reference_unet (UNet2DConditionModel): The UNet2DConditionModel used as a reference for inference.
        denoising_unet (UNet3DConditionModel): The UNet3DConditionModel used for denoising the input audio.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: EchoUNet3DConditionModel,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet

    def forward(self,):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
        }


def create_mask(mask_path, mask_512_path, clip_length):
        mask = 1 - (cv2.imread(mask_path) / 255.)
        (h, w, c) = mask.shape
        new_mask = np.zeros((h, w, c+1))
        new_mask[:, :, :c] = mask
        new_mask[:, :, c] = mask[:, :, c-1]

        mask = torch.FloatTensor(new_mask).permute(2,0,1).unsqueeze(1)

        return mask


def create_mask_alpha(w, h, coordinate):
    (left, top, right, bottom) = coordinate

    mask = np.ones((h,w,3))
    mask[top:bottom, left:right] = 0
    
    pad = min(int((right-left)/15), int((bottom-top)/15)) # 100/15=6.6% for padding
    
    for idx in range(left, right):
        mask[bottom-pad:bottom, idx] = np.array([np.linspace(0,1,pad)]*3).transpose(1,0)
        mask[top:top+pad, idx] = np.array([np.linspace(1,0,pad)]*3).transpose(1,0)
    
    for idx in range(top, bottom):
        mask[idx, right-pad:right] = np.array([np.linspace(0,1,pad)]*3).transpose(1,0)
        mask[idx, left:left+pad] = np.array([np.linspace(1,0,pad)]*3).transpose(1,0)
    
    for idx, start, end in zip(range(top, top+pad), np.linspace(1,0.7, pad), np.linspace(0.7,0, pad)):
        mask[idx, left:left+pad] = np.array([np.linspace(start,end,pad)]*3).transpose(1,0)
        mask[idx, right-pad:right] = np.array([np.linspace(end,start,pad)]*3).transpose(1,0)
    
    for idx, start, end in zip(range(bottom-pad, bottom), np.linspace(0.7,1, pad), np.linspace(0,0.7, pad)):
        mask[idx, left:left+pad] = np.array([np.linspace(start,end,pad)]*3).transpose(1,0)
        mask[idx, right-pad:right] = np.array([np.linspace(end,start,pad)]*3).transpose(1,0)
    
    return mask

def process_audio(input_audio_path, output_audio_path, device=None):
    wav,_ = librosa.load(input_audio_path, sr=16000)
    tmp_path = output_audio_path[:-4]+"_16000.wav"
    sf.write(tmp_path, wav, 16000)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pretrained.dns64().to(device)
    wav, sr = torchaudio.load(input_audio_path)
    wav = convert_audio(wav.to(device), sr, model.sample_rate, model.chin)

    with torch.no_grad():
        denoised = model(wav[None])[0]
        
    sf.write(output_audio_path, denoised.data.cpu().numpy()[0], 16000)

    return

img_size = (512,512)
    

def mem_gpu():
    # Total memory
    total = torch.cuda.get_device_properties(0).total_memory
    # Allocated memory
    allocated = torch.cuda.memory_allocated(0)
    # Cached memory (used for optimization)
    cached = torch.cuda.memory_reserved(0)

    print(f"Total: {total / 1e9:.2f} GB")
    print(f"Allocated: {allocated / 1e9:.2f} GB")
    print(f"Cached: {cached / 1e9:.2f} GB")


def prepare_inputs(source_video, driving_audio, save_path, fps):
    """Resample the video to the model fps and the audio to 16 kHz mono wav.
    Originally both had to be converted by hand before running."""
    stream = cv2.VideoCapture(str(source_video))
    source_fps = stream.get(cv2.CAP_PROP_FPS)
    stream.release()

    if abs(source_fps - fps) > 0.01:
        resampled_video = os.path.join(save_path, f"source_{fps}fps.mp4")
        print(f"resampling video from {source_fps:.2f} to {fps} fps")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source_video),
            "-r", str(fps), "-an", "-c:v", "libx264", "-crf", "16", resampled_video
        ], check=True)
        source_video = resampled_video

    audio_16k = os.path.join(save_path, "driving_audio_16k.wav")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(driving_audio),
        "-vn", "-ac", "1", "-ar", "16000", audio_16k
    ], check=True)

    return source_video, audio_16k


def default_output_name(config):
    """highsync_<audio>_<video>_s<steps>_cfg<cfg>_as<audio_scale>_<dtype>.mp4"""
    audio_name = Path(str(config.driving_audio)).stem
    video_name = Path(str(config.source_video)).stem
    return (f"highsync_{audio_name}_{video_name}"
            f"_s{config.inference_steps}_cfg{config.cfg_scale:g}"
            f"_as{config.audio_scale:g}"
            f"{'_cfgref' if config.cfg_mode == 'reference' else ''}"
            f"{f'_ov{config.context_overlap}' if config.context_overlap != 0 else ''}"
            f"_{config.weight_dtype}.mp4")


def log_run(csv_path, row):
    rows, fieldnames = [], list(row.keys())
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames += [f for f in (reader.fieldnames or []) if f not in fieldnames]
    rows.append(row)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inference_process(args: argparse.Namespace):
    """
    Perform inference processing.

    Args:
        args (argparse.Namespace): Command-line arguments.

    This function initializes the configuration for the inference process. It sets up the necessary
    modules and variables to prepare for the upcoming inference steps.
    """
    t_start = time.perf_counter()
    timings = {}

    # 1. init config
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)
    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if config.get("output") is None:
        output_dir = config.get("output_dir") or "./output"
        os.makedirs(output_dir, exist_ok=True)
        config.output = os.path.join(output_dir, default_output_name(config))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(str(config.output))), exist_ok=True)
    print(f"output: {config.output}")

    # 2. runtime variables
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32

    # 3. prepare inference data
    # 3.1 prepare source image, face mask
    img_size = (config.data.source_image.width,
                config.data.source_image.height)
    clip_length = config.data.n_sample_frames
    mask_path = config.mask_path
    mask_512_path = config.mask_512_path

    temp_dir = os.path.join(save_path, "img")
    os.makedirs(temp_dir, exist_ok=True)
    # shutil is platform independent and also handles paths containing spaces.
    for item in Path(temp_dir).iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # 3.2 prepare audio embeddings
    sample_rate = config.data.driving_audio.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps = config.data.export_video.fps

    t0 = time.perf_counter()
    source_video, audio_16k = prepare_inputs(
        config.source_video, config.driving_audio, save_path, fps)

    ### load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    audio_temp_path = os.path.join(save_path, "audio.wav")
    process_audio(audio_16k, audio_temp_path, device=device)
    timings["prepare_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()

    # 4. build modules
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    vae = AutoencoderKL.from_pretrained(config.vae.model_path)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.base_model_path, subfolder="unet")
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs)
    )

    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
    )

    # Freeze
    vae.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)

    # reference_unet.enable_gradient_checkpointing()
    # denoising_unet.enable_gradient_checkpointing()

    vae.eval()
    reference_unet.eval()
    denoising_unet.eval()

    net = Net(
        reference_unet,
        denoising_unet
    )


    # 5. inference
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        scheduler=val_noise_scheduler
    )


    pipeline.to(device=device, dtype=weight_dtype)
    timings["load_models_s"] = time.perf_counter() - t0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    image_processor = ImageProcessor(img_size)

    generator = torch.manual_seed(42)

    video_stream = cv2.VideoCapture(source_video)
    
    output_path = Path(str(config.output))
    temp_video_path = output_path.with_name(output_path.stem + "_no_audio.mp4")
    writer = imageio.get_writer(
    temp_video_path,
    fps=25,
    codec='libx264',
    quality=10,  # 0 (low) to 10 (high)
    ffmpeg_params=[
        '-pix_fmt', 'yuv420p',
        '-crf', '18'  # Lower CRF = better quality (17–23 is visually lossless range)
    ])
    
    whisper_feature = audio_processor.audio2feat(audio_temp_path)

    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
    # torch.save(whisper_chunks, "./cache/audio.pt")
    audio_frame_num = whisper_chunks.shape[0]
    audio_fea_final = torch.Tensor(whisper_chunks).to(dtype=vae.dtype, device=vae.device)
    # audio_fea_final = torch.tensor(torch.load("/workspace/preprocess/audio_emb_whisper/cM096XGOCG4_C1.pt")).to(dtype=vae.dtype, device=vae.device)
    audio_fea_final = audio_fea_final.unsqueeze(0)

    video_length = min(int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)), audio_frame_num)
    if video_length < audio_frame_num:
        audio_fea_final = audio_fea_final[:, :video_length, :, :]

    times = int(np.ceil(video_length / config.inference_frames_num))
    timings.update(face_detect_s=0.0, diffusion_s=0.0, composite_s=0.0)

    frames_video = []
    count_motion = 0
    count_img = 0
    count = 0
    finish = False
    
    while 1:

        still_reading, frame = video_stream.read()

        if (not still_reading) or (count_motion >= video_length):
            video_stream.release()
            finish = True
        else:
            frames_video.append(frame)
            count_motion += 1

        if len(frames_video)==0:
            break
        
        if ((count_motion % config.inference_frames_num) == 0) or finish:

            print(f"processing {count + 1}/{times}")
            t0 = time.perf_counter()
            source_image_pixels_all, bbox_all, frame_list_all = image_processor.preprocess_frames_jump(frames_video)
            timings["face_detect_s"] += time.perf_counter() - t0

            del frames_video
            
            for source_image_pixels_list, bbox_list, frame_list in zip(source_image_pixels_all, bbox_all, frame_list_all):

                print(f"generating {count + 1}/{times}")

                if bbox_list[0]!=(0,0,0,0):
                    
                    audio_tensor = audio_fea_final[:, count_img : count_img + len(frame_list)]

                    main_length = len(frame_list)
                    diff = 12 - main_length 

                    if diff > 0:
                        source_image_pixels_list.extend([source_image_pixels_list[-1]]*diff)
                        audio_tensor = torch.cat((audio_tensor, audio_tensor[:,-1:].repeat(1,diff,1,1)), dim=1)

                    source_image_pixels = torch.cat(source_image_pixels_list, dim=0)
                    mask_main = create_mask(mask_path, mask_512_path, len(frame_list))
                    mask_main = mask_main.unsqueeze(0).to(device=device, dtype=weight_dtype)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        pipeline_output = pipeline(
                            gt_encode=source_image_pixels,
                            audio_tensor=audio_tensor,
                            mask_main=mask_main,
                            width=img_size[0],
                            height=img_size[1],
                            video_length=len(source_image_pixels_list),
                            num_inference_steps=config.inference_steps,
                            guidance_scale=config.cfg_scale,
                            audio_feature_ratio=config.audio_scale,
                            cfg_mode=config.cfg_mode,
                            ref_cache=config.ref_cache == "on",
                            context_overlap=config.context_overlap,
                            generator=generator
                        )

                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    timings["diffusion_s"] += time.perf_counter() - t0
                    t0 = time.perf_counter()

                    images_out = pipeline_output.videos.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()[:main_length]  # convert to [f, h, w, c]
                    images_out = np.clip(images_out * 255, 0, 255).astype(np.uint8)  # to [0, 255]

                    for image_out, bbox, frame in zip(images_out, bbox_list, frame_list):
                        
                        frame_h, frame_w, _ = frame.shape
                        path = os.path.join(temp_dir, f"image_{count_img}.png")
                        
                        if bbox==(0,0,0,0):
                            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            # cv2.imwrite(path, frame)
                        else:
                            image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
                            frame_in_out = frame.copy()
                            frame_in_out[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cv2.resize(image_out, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
                            mask = create_mask_alpha(frame_w, frame_h, bbox)
                            blurred_image = (frame * mask + frame_in_out * (1-mask)).astype(np.uint8)
                            writer.append_data(cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))    
                            # cv2.imwrite(path, blurred_image)
                        
                        count_img += 1

                    timings["composite_s"] += time.perf_counter() - t0

                else:
                    for frame in frame_list:
                        path = os.path.join(temp_dir, f"image_{count_img}.png")
                        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        # cv2.imwrite(path, frame)
                        count_img += 1

            count += 1

            frames_video = []

        if finish:
            break

    writer.close()

    t0 = time.perf_counter()
    # original: re-encoded with -c:v h264; copy keeps the CRF 18 stream from the writer
    subprocess.run([
        "ffmpeg", "-i", str(temp_video_path), "-i", str(config.driving_audio),
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-y", str(config.output), "-loglevel", "error"
    ], check=True, text=True)
    Path(temp_video_path).unlink(missing_ok=True)
    timings["mux_s"] = time.perf_counter() - t0

    total_s = time.perf_counter() - t_start
    peak_vram_gib = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output": Path(str(config.output)).name,
        "video": Path(str(config.source_video)).name,
        "audio": Path(str(config.driving_audio)).name,
        "frames": count_img,
        "steps": config.inference_steps,
        "cfg": config.cfg_scale,
        "audio_scale": config.audio_scale,
        "cfg_mode": config.cfg_mode,
        "ref_cache": config.ref_cache,
        "context_overlap": config.context_overlap,
        "dtype": config.weight_dtype,
        **{k: round(v, 1) for k, v in timings.items()},
        "total_s": round(total_s, 1),
        "s_per_frame": round(total_s / max(count_img, 1), 3),
        "s_per_video_second": round(total_s / max(count_img / fps, 1e-6), 2),
        "peak_vram_gib": round(peak_vram_gib, 2),
    }
    print("timings: " + ", ".join(f"{k}={v}" for k, v in row.items()))
    log_run(os.path.join(os.path.dirname(os.path.abspath(str(config.output))), "highsync_runs.csv"), row)

    # output_file = config.output
    # frames_path = os.path.join(temp_dir, f"image_%d.png")
    # save_video(frames_path, config.driving_audio, output_file, frame_w, frame_h)

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default="configs/inference/default.yaml")
    parser.add_argument("--source_video", type=str, required=False, help="source image")
    parser.add_argument("--driving_audio", type=str, required=False, help="driving audio")
    parser.add_argument("--output", type=str, help="output video file name (default: auto-named in --output_dir)")
    parser.add_argument("--output_dir", type=str, help="directory for auto-named outputs", default="./output")
    parser.add_argument("--inference_frames_num", type=int, help="inference frames number", default=1500)
    parser.add_argument("--inference_steps", type=int, help="override config inference_steps")
    parser.add_argument("--cfg_scale", type=float, help="override config cfg_scale")
    parser.add_argument("--audio_scale", type=float,
                        help="strength of the audio cross-attention (original: hardcoded 3.0)")
    parser.add_argument("--cfg_mode", type=str, choices=["full", "reference"],
                        help="what the unconditional branch drops: reference+audio (full) or only the reference "
                             "(original: full)")
    parser.add_argument("--ref_cache", type=str, choices=["on", "off"],
                        help="compute reference UNet features once per frame instead of per window and step "
                             "(original: per window)")
    parser.add_argument("--context_overlap", type=int,
                        help="frames shared by neighbouring 12-frame context windows, default 0 (original: 3)")
    parser.add_argument("--weight_dtype", type=str, choices=["fp16", "bf16", "fp32"], help="override config weight_dtype")


    command_line_args = parser.parse_args()

    inference_process(command_line_args)
