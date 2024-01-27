import torch
from diffusers import PixArtAlphaPipeline
from transformers import T5EncoderModel
import gc

queue = []

queue.extend([{ 'prompt': 'Oppenheimer sits on the beach on a chair, watching a nuclear explosion with a huge mushroom cloud, 1200mm' }])

queue.extend([{
  'prompt': 'pirate ship trapped in a cosmic malestrom nebula',
  'width': 1024,
  'height': 1024,
  'seed': 1152753,
  'cfg': 5,
  'steps': 30,
}])

queue.extend([{ 'prompt': 'supercar', 'cfg': 4 } for _ in range(3)])

text_encoder = T5EncoderModel.from_pretrained(
  'PixArt-alpha/PixArt-XL-2-1024-MS',
  subfolder='text_encoder',
  torch_dtype=torch.float16,
  device_map='auto',
)

pipe = PixArtAlphaPipeline.from_pretrained(
  'PixArt-alpha/PixArt-XL-2-1024-MS',
  torch_dtype=torch.float16,
  text_encoder=text_encoder,
  transformer=None,
  device_map='auto',
)

with torch.no_grad():
  for generation in queue:
    generation['embeddings'] = pipe.encode_prompt(generation['prompt'])

del text_encoder
del pipe
gc.collect()
torch.cuda.empty_cache()

pipe = PixArtAlphaPipeline.from_pretrained(
  'PixArt-alpha/PixArt-XL-2-1024-MS',
  torch_dtype=torch.float16,
  text_encoder=None,
).to('cuda')

for i, generation in enumerate(queue, start=1):
  generator = torch.Generator(device='cuda')

  if 'seed' in generation:
    generator.manual_seed(generation['seed'])
  else:
    generator.seed()

  image = pipe(
    negative_prompt=None,
    width=generation['width'] if 'width' in generation else 1024,
    height=generation['height'] if 'height' in generation else 1024,
    guidance_scale=generation['cfg'] if 'cfg' in generation else 7,
    num_inference_steps=generation['steps'] if 'steps' in generation else 20,
    generator=generator,
    prompt_embeds=generation['embeddings'][0],
    prompt_attention_mask=generation['embeddings'][1],
    negative_prompt_embeds=generation['embeddings'][2],
    negative_prompt_attention_mask=generation['embeddings'][3],
    num_images_per_prompt=1,
  ).images[0]

  image.save(f'image_{i}.png')
