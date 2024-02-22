from time import perf_counter
import gc
import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoPipelineForText2Image

def encode_prompt(prompts, tokenizers, text_encoders):
  embeddings_list = []

  for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
    cond_input = tokenizer(
      prompt,
      max_length=tokenizer.model_max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt',
    )

    prompt_embeds = text_encoder(cond_input.input_ids.to('cuda'), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    embeddings_list.append(prompt_embeds.hidden_states[-2])

  prompt_embeds = torch.concat(embeddings_list, dim=-1)

  negative_prompt_embeds = torch.zeros_like(prompt_embeds)
  negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

  bs_embed, seq_len, _ = prompt_embeds.shape
  prompt_embeds = prompt_embeds.repeat(1, 1, 1)
  prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

  seq_len = negative_prompt_embeds.shape[1]
  negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
  negative_prompt_embeds = negative_prompt_embeds.view(1 * 1, seq_len, -1)

  pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
  negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)

  return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

queue = []

queue.extend([{
  'prompt': '3/4 shot, candid photograph of a beautiful 30 year old redhead woman with messy dark hair, peacefully sleeping in her bed, night, dark, light from window, dark shadows, masterpiece, uhd, moody',
  'seed': 877866765,
}])

queue.extend([{
  'prompt': 'futuristic living room with big windows, brown sofas, coffee table, plants, cyberpunk city, concept art, earthy colors',
  'seed': 5567822456,
}])

queue.extend([{
  'prompt': 'macro shot of a bee collecting nectar from lavender flowers',
  'seed': 2257899453,
}])

queue.extend([{
  'prompt': '3d rendered isometric fiji island beach, 3d tile, polygon, cartoony, mobile game',
  'seed': 987867834,
}])

tokenizer = CLIPTokenizer.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-1.0',
  subfolder='tokenizer',
)

text_encoder = CLIPTextModel.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-1.0',
  subfolder='text_encoder',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to('cuda')

tokenizer_2 = CLIPTokenizer.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-1.0',
  subfolder='tokenizer_2',
)

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-1.0',
  subfolder='text_encoder_2',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to('cuda')

with torch.no_grad():
  for generation in queue:
    image_start = perf_counter()

    generation['embeddings'] = encode_prompt(
      [generation['prompt'], generation['prompt']],
      [tokenizer, tokenizer_2],
      [text_encoder, text_encoder_2],
    )

    generation['total_time'] = perf_counter() - image_start

del tokenizer, text_encoder, tokenizer_2, text_encoder_2
gc.collect()
torch.cuda.empty_cache()

pipe = AutoPipelineForText2Image.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-1.0',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
  tokenizer=None,
  text_encoder=None,
  tokenizer_2=None,
  text_encoder_2=None,
).to('cuda')

generator = torch.Generator(device='cuda')

# warm up
for generation in queue:
  pipe(
    prompt_embeds=generation['embeddings'][0],
    negative_prompt_embeds =generation['embeddings'][1],
    pooled_prompt_embeds=generation['embeddings'][2],
    negative_pooled_prompt_embeds=generation['embeddings'][3],
    output_type='latent',
  )

for i, generation in enumerate(queue, start=1):
  image_start = perf_counter()

  generator.manual_seed(generation['seed'])

  generation['latents'] = pipe(
    prompt_embeds=generation['embeddings'][0],
    negative_prompt_embeds =generation['embeddings'][1],
    pooled_prompt_embeds=generation['embeddings'][2],
    negative_pooled_prompt_embeds=generation['embeddings'][3],
    generator=generator,
    output_type='latent',
  ).images

  generation['total_time'] = generation['total_time'] + (perf_counter() - image_start)

del pipe.unet
gc.collect()
torch.cuda.empty_cache()

pipe.upcast_vae()

with torch.no_grad():
  for i, generation in enumerate(queue, start=1):
    image_start = perf_counter()

    generation['latents'] = generation['latents'].to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

    image = pipe.vae.decode(
      generation['latents'] / pipe.vae.config.scaling_factor,
      return_dict=False,
    )[0]

    image = pipe.image_processor.postprocess(image, output_type='pil')[0]
    image.save(f'image_{i}.png')

    generation['total_time'] = generation['total_time'] + (perf_counter() - image_start)

images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
print('Image time:', images_totals)

images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
print('Average image time:', images_average)

max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
print('Max. memory used:', max_memory, 'GB')
