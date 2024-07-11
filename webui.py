'''
Things that I need to work on:
    - Defaults button

'''

import gradio as gr
import os
import torch
import time
import yaml

from utils import *

# Path to the settings file
SETTINGS_FILE_PATH = "Configs/generate_settings.yaml"
GENERATE_SETTINGS = {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_phonemizer = None
model = None
model_params = None
sampler = None
textcleaner = None
to_mel = None

def load_all_models(voice, model_path=None):
    global global_phonemizer, model, model_params, sampler, textcleaner, to_mel
    config = load_configurations(get_model_configuration(voice))
    if not model_path:
        model_path = load_voice_model(voice)
    sigma_value = config['model_params']['diffusion']['dist']['sigma_data']
    
    model, model_params = load_models_webui(sigma_value, device)
    global_phonemizer = load_phonemizer()
    
    sampler = create_sampler(model)
    textcleaner = TextCleaner()
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    
    load_pretrained_model(model, model_path=model_path)

def get_file_path(root_path, voice, file_extension, error_message):
    model_path = os.path.join(root_path, voice)
    if not os.path.exists(model_path):
        raise gr.Error(f'No {file_extension} located in "{root_path}" folder')

    for file in os.listdir(model_path):
        if file.endswith(file_extension):
            return os.path.join(model_path, file)
    
    raise gr.Error(error_message)

def get_model_configuration(voice):
    return get_file_path(root_path="Models", voice=voice, file_extension=".yml",error_message= "No configuration for Model specified located")

def load_voice_model(voice):
    return get_file_path(root_path="Models", voice=voice, file_extension=".pth", error_message="No TTS model found in specified location")

def generate_audio(text, voice, reference_audio_file, seed, alpha, beta, diffusion_steps, embedding_scale, voice_model, voices_root="voices",):
    original_seed = int(seed)
    reference_audio_path = os.path.join(voices_root, voice, reference_audio_file)
    reference_dicts = {f'{voice}': f"{reference_audio_path}"}
    # noise = torch.randn(1, 1, 256).to(device)
    start = time.time()
    if original_seed==-1:
        seed_value = random.randint(0, 2**32 - 1)
    else:
        seed_value = original_seed
    set_seeds(seed_value)
    for k, path in reference_dicts.items():
        mean, std = -4, 4
        ref_s = compute_style(path, model, to_mel, mean, std, device)
        
        wav1 = inference(text, ref_s, model, sampler, textcleaner, to_mel, device, model_params, global_phonemizer=global_phonemizer, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        rtf = (time.time() - start)
        print(f"RTF = {rtf:5f}")
        print(f"{k} Synthesized:")
        from scipy.io.wavfile import write
        os.makedirs("results", exist_ok=True)
        audio_opt_path = os.path.join("results", f"{voice}_output.wav")
        write(audio_opt_path, 24000, wav1)
    
    # Save the settings after generation
    save_settings({
        "text": text,
        "voice": voice,
        "reference_audio_file": reference_audio_file,
        "seed": original_seed if original_seed == -1 else seed_value,
        "alpha": alpha,
        "beta": beta,
        "diffusion_steps": diffusion_steps,
        "embedding_scale": embedding_scale,
        "voice_model" : voice_model
    })


    return audio_opt_path, [[seed_value]]

def train_model(data):
    return f"Model trained with data: {data}"

def update_settings(setting_value):
    return f"Settings updated to: {setting_value}"

def get_reference_audio_list(voice_name, root="voices"):
    reference_directory_list = os.listdir(os.path.join(root, voice_name))
    return reference_directory_list

def get_voice_models(voice):
    model_root = "Models"
    model_path = os.path.join(model_root,voice)
    model_list = [model_name for model_name in os.listdir(model_path) if model_name.endswith(".pth")]
    return model_list

def update_reference_audio(voice):
    return gr.Dropdown(choices=get_reference_audio_list(voice), value=get_reference_audio_list(voice)[0])

def update_voice_model(voice, model_name):
    gr.Info("Wait for models to load...")
    model_path = get_models_path(voice, model_name)
    load_all_models(voice=voice, model_path=model_path)
    gr.Info("Models finished loading")

def get_models_path(voice, model_name, root="Models"):
    return os.path.join(root, voice, model_name)

def update_voice_settings(voice):
    try:
        gr.Info("Wait for models to load...")
        model_name = get_voice_models(voice)    
        model_path = get_models_path(voice, model_name[0])   
        load_all_models(voice, model_path=model_path)
        ref_aud_path = update_reference_audio(voice)
        
        gr.Info("Models finished loading")
        return ref_aud_path, gr.Dropdown(choices=model_name, value=model_name[0])
    except:
        gr.Warning("No models found for the chosen voice chosen, new models not loaded")
        ref_aud_path = update_reference_audio(voice)
        return ref_aud_path, gr.Dropdown(choices=[]) 

def load_settings():
    try:
        with open(SETTINGS_FILE_PATH, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "text": "Hello there!",
            "voice": voice_list_with_defaults[0],
            "reference_audio_file": reference_audio_list[0],
            "seed" : "-1",
            "alpha": 0.3,
            "beta": 0.7,
            "diffusion_steps": 30,
            "embedding_scale": 1.0,
            "voice_model" : ""
        }

def save_settings(settings):
    with open(SETTINGS_FILE_PATH, "w") as f:
        yaml.safe_dump(settings, f)
        
def update_button_proxy():
    voice_list_with_defaults = get_voice_list(append_defaults=True)
    return gr.Dropdown(choices=voice_list_with_defaults)
        
voice_list_with_defaults = get_voice_list(append_defaults=True)
reference_audio_list = get_reference_audio_list(voice_list_with_defaults[0])


# Load models with the default or loaded settings
initial_settings = load_settings()
load_all_models(initial_settings["voice"])
list_of_models = get_voice_models(voice=initial_settings["voice"])

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Generation"):
            
            with gr.Column():
                with gr.Row():
                    GENERATE_SETTINGS["text"] = gr.Textbox(label="Input Text", value=initial_settings["text"])
                with gr.Row():
                    with gr.Column():
                        # Row 2: Existing content
                        GENERATE_SETTINGS["voice"] = gr.Dropdown(
                            choices=voice_list_with_defaults, label="Voice", type="value", value=initial_settings["voice"])
                        
                        GENERATE_SETTINGS["voice_model"] = gr.Dropdown(
                            choices=list_of_models, label="Voice Models", type="value", value=initial_settings["voice_model"])
                        
                        GENERATE_SETTINGS["reference_audio_file"] = gr.Dropdown(
                            choices=get_reference_audio_list(initial_settings["voice"]), label="Reference Audio", type="value", value=initial_settings["reference_audio_file"]
                        )
                    with gr.Column():
                        GENERATE_SETTINGS["seed"] = gr.Textbox(
                            label="Seed", value=initial_settings["seed"]
                        )
                        GENERATE_SETTINGS["alpha"] = gr.Slider(
                            label="alpha", minimum=0, maximum=2.0, step=0.1, value=initial_settings["alpha"]
                        )
                        GENERATE_SETTINGS["beta"] = gr.Slider(
                            label="beta", minimum=0, maximum=2.0, step=0.1, value=initial_settings["beta"]
                        )
                        GENERATE_SETTINGS["diffusion_steps"] = gr.Slider(
                            label="Diffusion Steps", minimum=0, maximum=400, step=1, value=initial_settings["diffusion_steps"]
                        )
                        GENERATE_SETTINGS["embedding_scale"] = gr.Slider(
                            label="Embedding Scale", minimum=0, maximum=4.0, step=0.1, value=initial_settings["embedding_scale"]
                        )
                    with gr.Column():
                        generation_output = gr.Audio(label="Output")
                        seed_output = gr.Dataframe(
                            headers=["Seed"], 
                            datatype=["number"],
                            value=[], 
                            height=200,  
                            min_width=200  
                        )
                with gr.Row():
                    update_button = gr.Button("Update Voices")
                    generate_button = gr.Button("Generate")
                
                # Clicks
        
                generate_button.click(generate_audio, 
                                    inputs=[GENERATE_SETTINGS["text"],
                                            GENERATE_SETTINGS["voice"],
                                            GENERATE_SETTINGS["reference_audio_file"],
                                            GENERATE_SETTINGS["seed"],
                                            GENERATE_SETTINGS["alpha"],
                                            GENERATE_SETTINGS["beta"],
                                            GENERATE_SETTINGS["diffusion_steps"],
                                            GENERATE_SETTINGS["embedding_scale"],
                                            GENERATE_SETTINGS["voice_model"]], 
                                    outputs=[generation_output, seed_output])
                
                update_button.click(update_button_proxy,
                                    outputs=GENERATE_SETTINGS["voice"])
                
                # Changes
                GENERATE_SETTINGS["voice"].change(fn=update_voice_settings, 
                            inputs=GENERATE_SETTINGS["voice"], 
                            outputs=[GENERATE_SETTINGS["reference_audio_file"],
                                     GENERATE_SETTINGS["voice_model"]])
                
                GENERATE_SETTINGS["voice_model"].change(fn=update_voice_model,
                            inputs=[GENERATE_SETTINGS["voice"],
                                    GENERATE_SETTINGS["voice_model"]])
        
        with gr.TabItem("Training"):
            with gr.Tabs():
                with gr.TabItem("Prepare Dataset"):
                    pass
                
                with gr.TabItem("Generate Configuration"):
                    pass
                
                with gr.TabItem("Run Training"):
                    pass
        
        with gr.TabItem("Settings"):
            settings_input = gr.Textbox(label="Enter setting value")
            settings_output = gr.Textbox(label="Settings Output", interactive=False)
            settings_button = gr.Button("Update Settings")
            settings_button.click(update_settings, inputs=settings_input, outputs=settings_output)

# Launch the interface
demo.launch()
