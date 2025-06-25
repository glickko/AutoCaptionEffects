import sys
import os
import re
import subprocess
import json
import time
import logging
import webview
import threading
import shutil

# Import winreg only on Windows
if sys.platform == 'win32':
    import winreg

# Import torch safely at the top
try:
    import torch
except ImportError:
    pass

from datetime import timedelta
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, ColorClip, ImageClip, AudioFileClip
from PIL import Image, ImageColor
import cv2
import moviepy.config as cfg

# Import the new transcription library
import stable_whisper

# Patch for Pillow 10+
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s:%(message)s",
                    handlers=[logging.FileHandler("subtitle_app.log"),
                              logging.StreamHandler()])

# --- Global Variables & Setup ---
window = None
IMAGEMAGICK_PATH = None
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
TEMP_DIR_BASE = os.path.join(os.getcwd(), 'temp')
SETTINGS_FILE = os.path.join(os.getcwd(), 'settings.json')


# --- Auto-Detection and Utility Functions ---

def find_imagemagick_path_windows():
    if sys.platform != 'win32':
        return None
    try:
        key_path = r"SOFTWARE\ImageMagick\Current"
        for root_key in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
            try:
                with winreg.OpenKey(root_key, key_path, 0, winreg.KEY_READ) as key:
                    bin_path, _ = winreg.QueryValueEx(key, "BinPath")
                    if bin_path and os.path.exists(os.path.join(bin_path, "magick.exe")):
                        magick_exe_path = os.path.join(bin_path, "magick.exe")
                        logging.info(f"Auto-detected ImageMagick at: {magick_exe_path}")
                        return magick_exe_path
            except FileNotFoundError:
                continue
    except Exception as e:
        logging.warning(f"Could not auto-detect ImageMagick from registry: {e}")
    return None

def get_imagemagick_fonts():
    global IMAGEMAGICK_PATH
    if IMAGEMAGICK_PATH is None:
        return ["Helvetica-Bold", "Arial", "Times-New-Roman", "Courier"]
    try:
        output = subprocess.check_output([IMAGEMAGICK_PATH, "-list", "font"], stderr=subprocess.STDOUT).decode("utf-8")
        fonts = [line.split(":", 1)[1].strip() for line in output.splitlines() if line.strip().startswith("Font:")]
        return fonts if fonts else ["Helvetica-Bold", "Arial", "Times-New-Roman"]
    except Exception as e:
        logging.error("Error getting fonts from ImageMagick: %s", e)
        return ["Helvetica-Bold", "Arial", "Times-New-Roman"]

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def create_rounded_overlay(size, color, radius=10, duration=1):
    w, h = int(size[0]), int(size[1])
    radius = min(radius, w // 2, h // 2)
    image = np.zeros((h, w, 4), dtype=np.uint8)
    if isinstance(color, str):
        rgb_color = ImageColor.getrgb(color)
    else:
        rgb_color = color
    cv2.rectangle(image, (radius, 0), (w - radius, h), (*rgb_color, 255), -1)
    cv2.rectangle(image, (0, radius), (w, h - radius), (*rgb_color, 255), -1)
    cv2.circle(image, (radius, radius), radius, (*rgb_color, 255), -1)
    cv2.circle(image, (w - radius, radius), radius, (*rgb_color, 255), -1)
    cv2.circle(image, (radius, h - radius), radius, (*rgb_color, 255), -1)
    cv2.circle(image, (w - radius, h - radius), radius, (*rgb_color, 255), -1)
    overlay = ImageClip(image, duration=duration)
    return overlay
    
def export_srt_file(result, srt_path):
    try:
        result.to_srt_vtt(srt_path, word_level=False)
        logging.info(f"SRT file successfully exported to {srt_path}")
    except Exception as e:
        logging.error(f"Failed to export SRT file: {e}")

def extract_audio(input_path, audio_path, ffmpeg_threads=4):
    try:
        command = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", audio_path]
        if ffmpeg_threads > 0:
            command.insert(2, "-threads")
            command.insert(3, str(ffmpeg_threads))
        subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        return audio_path
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode()
        logging.error("FFmpeg extraction error: %s", error_message)
        window.evaluate_js(f'showError("Gagal mengekstrak audio: {json.dumps(error_message)}")')
        return None
    except Exception as e:
        logging.error("Unexpected error during audio extraction: %s", e)
        window.evaluate_js(f'showError("Error saat ekstraksi audio: {json.dumps(str(e))}")')
        return None

def create_text_with_outer_stroke(text, font, fontsize, color, stroke_color, stroke_width):
    main_text_for_size = TextClip(text, font=font, fontsize=fontsize, color=color, align="center")
    w, h = main_text_for_size.size
    canvas_w = w + stroke_width * 4
    canvas_h = h + stroke_width * 4
    main_text = TextClip(text, font=font, fontsize=fontsize, color=color, align="center").set_position(('center', 'center'))
    stroke_text = TextClip(text, font=font, fontsize=fontsize, color=stroke_color, align="center")
    stroke_clips = []
    for dx in [-stroke_width, 0, stroke_width]:
        for dy in [-stroke_width, 0, stroke_width]:
            if dx == 0 and dy == 0:
                continue
            stroke_clips.append(stroke_text.copy().set_position(('center', 'center')).set_position((dx, dy)))
    return CompositeVideoClip([*stroke_clips, main_text], size=(canvas_w, canvas_h))


# --- Core Logic with stable_whisper ---

def transcribe_audio(params, update_callback):
    audio_path = params['audio_path']
    model_size = params['model_size']
    language = params['language'] if params['language'] != 'auto' else None
    device = "cuda" if 'torch' in sys.modules and torch.cuda.is_available() else "cpu"
    try:
        update_callback(15, "Loading transcription model...")
        model = stable_whisper.load_model(model_size, device=device)
        update_callback(30, f"Transcribing with '{model_size}' model...")
        result = model.transcribe(audio_path, language=language, fp16=(device=='cuda'), only_voice_freq=True)
        if params.get('refine_timestamps', False):
            update_callback(60, "Refining timestamps...")
            model.refine(audio_path, result)
        update_callback(75, "Applying advanced line splitting rules...")
        
        if params.get('split_by_punctuation', False):
            result.split_by_punctuation([('.', ' '), '。', '?', '？', ',', '，'])
            
        if params.get('split_by_gap', False):
            result.split_by_gap(params.get('gap_value', 0.4))
        
        result.merge_by_gap(params.get('merge_gap_value', 0.15), max_words=3)
        result.split_by_length(max_chars=int(params.get('max_chars', 25)))

        censor_list = [word.strip() for word in params.get('censor_list', '').split(',') if word.strip()]
        if censor_list:
            def censor_word(word_obj):
                match = word_obj.word.strip()
                word_obj.word = word_obj.word.replace(match, '*' * len(match))
                return word_obj
            result.custom_mapping(key=lambda word, **_: word.word.lower().strip('.,!?'), value=censor_list, method=censor_word)
        
        return result
    except Exception as e:
        logging.error("Error during transcription: %s", e)
        window.evaluate_js(f'showError("Transkripsi audio gagal: {json.dumps(str(e))}")')
        return None

def create_caption(segment, framesize, params):
    font, fontsize, color = params['font'], params['fontsize'], params['font_color']
    highlight_color, preset = params['highlight_color'], params['preset']
    highlight_opacity = params.get('highlight_opacity', 1.0)
    stroke_enabled = params.get('stroke_enabled', False)
    stroke_color, stroke_width = params.get('stroke_color', 'black'), params.get('stroke_width', 2)
    frame_width, frame_height = framesize
    
    full_text = segment.text.strip()
    format_option = params.get('format_text', 'none')
    if format_option == 'uppercase':
        full_text = full_text.upper()
    elif format_option == 'lowercase':
        full_text = full_text.lower()

    if params.get('remove_punctuation', False):
        full_text = re.sub(r'[^\w\s]', '', full_text)

    words = segment.words

    def make_clip(txt):
        if stroke_enabled:
            return create_text_with_outer_stroke(txt, font, fontsize, color, stroke_color, stroke_width)
        else:
            return TextClip(txt, font=font, fontsize=fontsize, color=color, align="center")

    def make_highlight_clip(txt):
        if stroke_enabled:
            return create_text_with_outer_stroke(txt, font, fontsize, highlight_color, stroke_color, stroke_width)
        else:
            return TextClip(txt, font=font, fontsize=fontsize, color=highlight_color, align="center")

    full_clip = make_clip(full_text)
    text_w, text_h = full_clip.size
    clips, margin = [], frame_height * 0.1
    y_pos = margin if params['placement'] == 'top' else frame_height - text_h - margin if params['placement'] == 'bottom' else (frame_height - text_h) / 2
    x_pos = (frame_width - text_w) / 2

    if preset == "Highlight" and words:
        highlight_style = params.get('highlight_style', 'Rectangle')
        highlight_padding = params.get('highlight_padding', 0)
        highlight_movement = params.get('highlight_movement', False)
        move = (lambda t: (3 * np.sin(2 * np.pi * t/3), 5 * np.sin(2 * np.pi * t/2))) if highlight_movement else (lambda t: (0, 0))
        text_clip = make_clip(full_text)
        text_clip = text_clip.set_position(lambda t: (x_pos + move(t)[0], y_pos + move(t)[1])).set_start(segment.start).set_duration(segment.duration)
        
        highlight_clips = []
        word_positions = []
        current_x_offset = 0
        space_clip = TextClip(" ", font=font, fontsize=fontsize)
        
        for word in words:
            word_text = word.word.strip()
            if format_option == 'uppercase': word_text = word_text.upper()
            elif format_option == 'lowercase': word_text = word_text.lower()
            if params.get('remove_punctuation', False): word_text = re.sub(r'[^\w\s]', '', word_text)
            
            word_size_clip = make_clip(word_text)
            word_positions.append(current_x_offset)
            current_x_offset += word_size_clip.size[0] + space_clip.size[0]

        for i, word in enumerate(words):
            word_duration = word.end - word.start
            if word_duration < 0.1: word_duration = 0.1
            
            word_text = word.word.strip()
            if format_option == 'uppercase': word_text = word_text.upper()
            elif format_option == 'lowercase': word_text = word_text.lower()
            if params.get('remove_punctuation', False): word_text = re.sub(r'[^\w\s]', '', word_text)

            word_size_clip = make_clip(word_text)
            padded_size = (word_size_clip.size[0] + 2 * highlight_padding, word_size_clip.size[1] + 2 * highlight_padding)
            
            if highlight_style == 'Rounded Rectangle':
                highlight_shape = create_rounded_overlay(padded_size, color=highlight_color, radius=10, duration=word_duration)
            else:
                highlight_shape = ColorClip(size=padded_size, color=highlight_color, duration=word_duration)

            # ** FADE ANIMATION ADDED **
            highlight_shape = highlight_shape.set_opacity(highlight_opacity).fadein(0.1).fadeout(0.1)
            
            shape_pos = (x_pos + word_positions[i] - highlight_padding, y_pos - highlight_padding)
            highlight_shape = highlight_shape.set_start(word.start).set_position(lambda t, sp=shape_pos: (sp[0] + move(t)[0], sp[1] + move(t)[1]))
            highlight_clips.append(highlight_shape)
            
        clips.extend(highlight_clips)
        clips.append(text_clip)

    elif preset == "Karaoke" and words:
        base_text = make_clip(full_text).set_position((x_pos, y_pos)).set_start(segment.start).set_duration(segment.duration)
        clips.append(base_text)
        for i, word in enumerate(words):
            karaoke_text_words = []
            for w in words[:i+1]:
                word_text = w.word
                if format_option == 'uppercase': word_text = word_text.upper()
                elif format_option == 'lowercase': word_text = word_text.lower()
                if params.get('remove_punctuation', False): word_text = re.sub(r'[^\w\s]', '', word_text)
                karaoke_text_words.append(word_text)
            karaoke_text = ' '.join(karaoke_text_words)
            highlighted_part = make_highlight_clip(karaoke_text)
            start_time, end_time = words[i].start, (segment.end if i == len(words) - 1 else words[i+1].start)
            duration = end_time - start_time
            if duration > 0: clips.append(highlighted_part.set_start(start_time).set_duration(duration).set_position((x_pos, y_pos)))
    
    else:
        main_clip = make_clip(full_text).set_position((x_pos, y_pos)).set_start(segment.start).set_duration(segment.duration)
        if preset == "Fade In": main_clip = main_clip.fadein(params.get('fadein_duration', 0.3))
        elif preset == "Pop In":
            from moviepy.video.fx.all import resize
            scale_duration = params.get('popin_duration', 0.6)
            scaling = lambda t: 0.5 + 0.5 * min(t / scale_duration, 1)
            main_clip = main_clip.fx(resize, scaling)
            orig_size = main_clip.size
            main_clip = main_clip.set_position(lambda t, w=orig_size[0], h=orig_size[1]: (x_pos + (w * (1 - scaling(t))) / 2, y_pos + (h * (1 - scaling(t))) / 2))
        clips.append(main_clip)
    return clips

def create_audiogram(params, result, update_callback):
    input_video_path = params['input_file']
    output_video = params['output_video_path']
    framesize = (params['output_width'], params['output_height'])
    update_callback(85, "Creating text clips for video...")
    all_text_clips = []
    fontsize_val = params.get('font_size_override', 'Default')
    params['fontsize'] = int(fontsize_val) if fontsize_val != 'Default' else min(max(24, int(framesize[1] / 20)), 40)
    for seg in result.segments: all_text_clips.extend(create_caption(seg, framesize, params))
    update_callback(90, "Compositing video...")
    ext = os.path.splitext(input_video_path)[1].lower()
    is_video = ext in [".mp4", ".mkv", ".avi", ".mov", ".flv"]

    if is_video:
        background = VideoFileClip(input_video_path).resize(newsize=framesize)
        if not hasattr(background, 'audio') or background.audio is None: background = background.set_audio(AudioFileClip(params['audio_path']))
    else:
        duration = result.segments[-1].end if result.segments else 10
        background = ColorClip(size=framesize, color=(0,0,0), duration=duration).set_audio(AudioFileClip(input_video_path))
        
    final_video = CompositeVideoClip([background] + all_text_clips, size=framesize)
    
    try:
        logging.info(f"Writing final video to: {output_video}")
        ffmpeg_params = ["-crf", "0"] if params.get('lossless_quality', False) else []
        final_video.write_videofile(output_video, fps=params.get('output_fps', 24), codec="libx264", audio_codec="aac", threads=params.get('ffmpeg_threads', 4), preset="veryslow" if params.get('lossless_quality', False) else 'medium', ffmpeg_params=ffmpeg_params)
        update_callback(100, "Audiogram created successfully!")
    except Exception as e:
        logging.error("Error writing video file: %s", e)
        window.evaluate_js(f'showError("Gagal membuat video: {json.dumps(str(e))}")')

def main_processing_thread(params):
    def update_progress(percentage, message): window.evaluate_js(f'updateProgress({percentage}, "{message}")')
    
    process_temp_dir = None
    try:
        start_time = time.time()
        input_file = params['input_file']
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        process_temp_dir = os.path.join(TEMP_DIR_BASE, f"process_{timestamp}")
        os.makedirs(process_temp_dir, exist_ok=True)

        base, ext = os.path.splitext(os.path.basename(input_file))
        subfolder_name = sanitize_filename(f"{base}_{timestamp}")
        process_output_dir = os.path.join(OUTPUT_DIR, subfolder_name)
        os.makedirs(process_output_dir, exist_ok=True)
        
        params['output_video_path'] = os.path.join(process_output_dir, f"{base}_subtitled.mp4")

        is_video = os.path.splitext(input_file)[1].lower() in [".mp4", ".mkv", ".avi", ".mov", ".flv"]
        temp_audio_path = os.path.join(process_temp_dir, "temp_audio.wav")
        audio_path = extract_audio(input_file, temp_audio_path, params.get('ffmpeg_threads', 4)) if is_video else input_file
        
        if not audio_path:
            update_progress(0, "Error: Audio extraction failed.")
            return
            
        params['audio_path'] = audio_path
        result = transcribe_audio(params, update_progress)
        
        if result:
            create_audiogram(params, result, update_progress)
            
            if params.get('export_srt', False):
                srt_path = os.path.join(process_output_dir, f"{base}.srt")
                export_srt_file(result, srt_path)
        else:
            update_progress(0, "Error: Transcription failed.")
            
        logging.info("Process completed in %.2f seconds", time.time() - start_time)
        
    finally:
        if process_temp_dir and os.path.exists(process_temp_dir):
            try:
                shutil.rmtree(process_temp_dir)
                logging.info(f"Successfully cleaned up temp folder: {process_temp_dir}")
            except Exception as e:
                logging.error(f"Failed to clean up temp folder {process_temp_dir}: {e}")

class Api:
    def open_file_dialog(self):
        file_types = ('Media Files (*.mp4;*.mkv;*.mov;*.flv;*.mp3;*.wav)', 'All files (*.*)')
        result = window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)
        if result:
            window.evaluate_js(f"document.getElementById('input_file').value = {json.dumps(result[0])};")
        
    def process_media(self, params):
        logging.info("Received process request with params: %s", params)
        if not params.get('input_file'):
            window.evaluate_js('showError("Please select an input media file.")')
            return
        threading.Thread(target=main_processing_thread, args=(params,)).start()

    def get_fonts(self): return get_imagemagick_fonts()

    def save_settings(self, settings):
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            logging.info("Settings saved successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
            return False

    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                logging.info("Settings loaded successfully.")
                return settings
        except Exception as e:
            logging.error(f"Failed to load settings: {e}")
        return {}


if __name__ == '__main__':
    if 'torch' not in sys.modules:
        logging.error("PyTorch is not installed. Please install it with 'pip install torch'.")
        sys.exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR_BASE, exist_ok=True)
    
    IMAGEMAGICK_PATH = find_imagemagick_path_windows()
    if IMAGEMAGICK_PATH: cfg.change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_PATH})
    else: logging.warning("ImageMagick path not found. Font selection will be limited.")
    api = Api()
    window = webview.create_window('AI Auto Subtitle Effects', 'frontend/index.html', js_api=api, width=960, height=540, resizable=True, min_size=(800, 500))
    webview.start()