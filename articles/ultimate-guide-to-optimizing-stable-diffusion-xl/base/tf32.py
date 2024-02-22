from time import perf_counter
import torch
from diffusers import AutoPipelineForText2Image

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

torch.backends.cudnn.allow_tf32 = False

pipe = AutoPipelineForText2Image.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-1.0',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to('cuda')

generator = torch.Generator(device='cuda')

# warm up
for generation in queue:
  pipe(generation['prompt'])

for i, generation in enumerate(queue, start=1):
  image_start = perf_counter()

  generator.manual_seed(generation['seed'])

  image = pipe(
    prompt=generation['prompt'],
    generator=generator,
  ).images[0]

  image.save(f'image_{i}.png')

  generation['total_time'] = perf_counter() - image_start

images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
print('Image time:', images_totals)

images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
print('Average image time:', images_average)

max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
print('Max. memory used:', max_memory, 'GB')
