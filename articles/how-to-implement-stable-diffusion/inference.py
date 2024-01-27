import torch
from torchvision.transforms import ToPILImage
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from tqdm.auto import tqdm
from PIL import Image

tokenizer = CLIPTokenizer.from_pretrained(
  'runwayml/stable-diffusion-v1-5',
  subfolder='tokenizer',
)

text_encoder = CLIPTextModel.from_pretrained(
  'runwayml/stable-diffusion-v1-5',
  subfolder='text_encoder',
  use_safetensors=True,
).to('cuda')

scheduler = EulerDiscreteScheduler.from_pretrained(
  'runwayml/stable-diffusion-v1-5',
  subfolder='scheduler',
)

unet = UNet2DConditionModel.from_pretrained(
  'runwayml/stable-diffusion-v1-5',
  subfolder='unet',
  use_safetensors=True,
).to('cuda')

vae = AutoencoderKL.from_pretrained(
  'runwayml/stable-diffusion-v1-5',
  subfolder='vae',
  use_safetensors=True,
).to('cuda')

prompts = ['silly dog wearing a batman costume, funny, realistic, canon, award winning photography']
batch_size = 1
inference_steps = 30
seed = 1055747
cfg_scale = 7
height = 512
width = 512

cond_input = tokenizer(
  prompts,
  max_length=tokenizer.model_max_length,
  padding='max_length',
  truncation=True,
  return_tensors='pt',
)

with torch.no_grad():
  cond_embeddings = text_encoder(cond_input.input_ids.to('cuda'))[0]

uncond_input = tokenizer(
  [''] * batch_size,
  max_length=tokenizer.model_max_length,
  padding='max_length',
  truncation=True,
  return_tensors='pt',
)

with torch.no_grad():
  uncond_embeddings = text_encoder(uncond_input.input_ids.to('cuda'))[0]

text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

generator = torch.Generator(device='cuda')
generator.manual_seed(seed)

latents = torch.randn(
  (batch_size, unet.config.in_channels, height // 8, width // 8),
  generator=generator,
  device='cuda',
)

scheduler.set_timesteps(inference_steps)

latents = latents * scheduler.init_noise_sigma

for t in tqdm(scheduler.timesteps):
  latent_model_input = torch.cat([latents] * 2)
  latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

  with torch.no_grad():
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

  noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
  noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

  latents = scheduler.step(noise_pred, t, latents).prev_sample

latents = latents / vae.config.scale_factor
with torch.no_grad():
  images = vae.decode(latents).sample

images = (images / 2 + 0.5).clamp(0, 1)

to_pil = ToPILImage()

for i in range(1, batch_size + 1):
  image = to_pil(images[i])
  image.save(f'image_{i}.png')
