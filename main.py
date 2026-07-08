GOOGLE_API_KEY="AQ.Ab8RN6Ki4yAEQZm0Xw0KnqxjW62JEp8yvUByCqmC9W5srPAQGg"

from google import genai
from google.genai import types

import json
from PIL import Image

import requests

from pydantic import BaseModel

import time
import io
import base64

class Prompt(BaseModel):
    name: str
    prompt: str

IMAGE_MODEL_ID = "gemini-2.5-flash-image"  # @param ["gemini-2.5-flash-image", "gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview"] {"allow-input":true, isTemplate: true}
GEMINI_MODEL_ID = "gemini-3.5-flash" # @param ["gemini-2.5-flash", "gemini-3.1-flash-lite-preview", "gemini-3.5-flash", "gemini-3.1-pro-preview"] {"allow-input":true, isTemplate: true}

# Models for optional paid features (Veo, Lyria, TTS)
LYRIA_MODEL_ID = "lyria-3-clip-preview" # @param ["lyria-3-clip-preview"] {"allow-input":true, isTemplate: true}
TTS_MODEL_ID = "gemini-3.1-flash-tts-preview" # @param ["gemini-2.5-flash-preview-tts","gemini-2.5-pro-preview-tts","gemini-3.1-flash-tts-preview"] {"allow-input":true, isTemplate: true}

VEO_MODEL_ID = "veo-3.1-lite-generate-preview" # @param ["veo-3.1-lite-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-generate-preview"] {"allow-input":true, isTemplate: true}

# These optional sections are all paid - toggle them on if you want to run them
I_understand_this_is_a_paid_API = True # @param {type:"boolean"}
service_tier = "flex" # @param ["flex","standard","priority"]

max_character_images = 5 # @param {type:"integer",isTemplate: true, min:1}
max_chapter_images = 3 # @param {type:"integer",isTemplate: true, min:1}

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=5,
            initial_delay=2.0,
            max_delay=60.0,
            http_status_codes=[429, 500, 502, 503, 504]
        )
    )
)

print("Downloading book...")
url = "https://www.gutenberg.org/cache/epub/730/pg730.txt"  # @param {type:"string"}

response = requests.get(url)
with open("book.txt", "wb") as file:
    file.write(response.content)

print("Book downloaded.")

book = client.files.upload(file="book.txt")

print("Book uploaded to Gemini.")

system_instructions = """
  There must be no text on the image, it should not look like a cover page.
  It should be an full illustration with no borders, titles, nor description.
  Unless asked otherwise, stay family-friendly with uplifting colors.
  Each produced should be a simple image, no panels.
"""

print("Creating initial interaction...")
# Start the conversation with the book content
book_interaction = client.interactions.create(
    model=GEMINI_MODEL_ID,
    input=[
        {"type": "text", "text": "Here's a book, to illustrate using Nano Banana. Don't say anything for now, instructions will follow."},
        {"type": "document", "uri": book.uri},
    ],
    service_tier=service_tier,
)
print("Initial interaction created.")

style = "comic book" # @param {type:"string", "placeholder":"Write your own style or leave empty to let Gemini generate one"}

print("Defining art style...")
if style=="":
  style_interaction = client.interactions.create(
      model=GEMINI_MODEL_ID,
      input="Can you define a art style that would fit the story but with a twist? Just give us the prompt for the art syle that will added to the furture prompts.",
      previous_interaction_id=book_interaction.id,
      service_tier=service_tier,
  )
  last_interaction = style_interaction
  style = style_interaction.output_text
else:
  style_interaction = client.interactions.create(
      model=GEMINI_MODEL_ID,
      input=f'The art style will be:"{style}". Keep that in mind when generating future prompts. Keep quiet for now, instructions will follow.',
      previous_interaction_id=book_interaction.id,
      service_tier=service_tier,
  )
  last_interaction = style_interaction

print("Art style defined: ", style)

style = f'Follow this style: "{style}" '

print("Creating characters prompts interaction...")
characters_prompts_interaction = client.interactions.create(
    model=GEMINI_MODEL_ID,
    input="Can you describe the main characters and prepare a prompt describing them with as much details as possible (use the descriptions from the book) so Nano Banana can generate images of them? Each prompt should be at least 50 words.",
    previous_interaction_id=style_interaction.id,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": {"type": "array", "items": Prompt.model_json_schema()},
    },
    service_tier=service_tier,
)
last_interaction = characters_prompts_interaction
print("Characters prompts interaction created.")
characters = json.loads(characters_prompts_interaction.output_text)

print("Characters: \n", json.dumps(characters, indent=4))

print("Creating characters image interaction...")
character_images = []
last_image_interaction = None

# Set up the image generation context
characters_image_interaction = client.interactions.create(
    model=IMAGE_MODEL_ID,
    input=f"""
      You are going to generate portrait images to illustrate The Wind in the Willows from Kenneth Grahame.
      The style we want you to follow is: {style}
      Also follow those rules: {system_instructions}
    """,
    service_tier=service_tier,
)
print("Characters image interaction created.")

for character in characters[:max_character_images]:
  print("Creating image for character: ", character['name'])

  characters_image_interaction = client.interactions.create(
      model=IMAGE_MODEL_ID,
      input=f"Create an illustration for {character['name']} following this description: {character['prompt']}",
      previous_interaction_id=characters_image_interaction.id,
      service_tier=service_tier,
  )

  # Extract image from interaction steps
  generated_image = None
  for step in reversed(characters_image_interaction.steps):
      if step.type == "model_output" and step.content:
          for content in reversed(step.content):
              if content.type == "image":
                  generated_image = content
                  break
          if generated_image:
              break

  if generated_image:
      print("Image generated for character: ", character['name'])
      // Save image to file.
      with open(f"{character['name']}.png", "wb") as file:
        file.write(generated_image.data)
      print("Image saved to file: ", f"{character['name']}.png")
  else:
      print(f"No image generated for {character['name']}")
  character_images.append(generated_image)
last_image_interaction = characters_image_interaction
print("Characters image interaction completed.")

print("Creating chapters prompts interaction...")
chapters_prompts_interaction = client.interactions.create(
    model=GEMINI_MODEL_ID,
    input="Now, for each chapters of the book, give me a prompt to illustrate what happens in it. It should be a single image, not a multi-tiled page. Be very descriptive, especially of the characters. Be very descriptive and remember to tell their name and to reuse the character prompts if they appear in the images. Also list all characters who appear in it.",
    previous_interaction_id=characters_prompts_interaction.id,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": {"type": "array", "items": Prompt.model_json_schema()},
    },
    service_tier=service_tier,
)
last_interaction = chapters_prompts_interaction
print("Chapters prompts interaction created.")
chapters = json.loads(chapters_prompts_interaction.steps[-1].content[0].text)[:max_chapter_images]
print("Chapters: \n", json.dumps(chapters, indent=4))

print("Creating chapters image interaction...")
chapter_images = []
chapters_image_interaction = client.interactions.create(
    model=IMAGE_MODEL_ID,
    input="Starting from now, we're going to illustrate the book's chapters. Don't forget to refer to your previous illustrations of the characters to keep the characters consistency, but feel free to change their position.",
    previous_interaction_id=last_image_interaction.id,
    service_tier=service_tier,
)
last_image_interaction = chapters_image_interaction
print("Chapters image interaction created.")

for chapter in chapters:
  print("Creating image for chapter: ", chapter['name'])

  chapters_image_interaction = client.interactions.create(
      model=IMAGE_MODEL_ID,
      input=f"Create an illustration for {chapter['name']} using the previously generated characters following this description: {chapter['prompt']}",
      previous_interaction_id=last_image_interaction.id,
      service_tier=service_tier,
  )
  last_image_interaction = chapters_image_interaction
  print("Chapter image interaction created.")

  for step in reversed(chapters_image_interaction.steps):
      if step.type == "model_output" and step.content:
          for content in reversed(step.content):
              if content.type == "image":
                  generated_image = content
                  // Save image to file.
                  with open(f"{chapter['name']}.png", "wb") as file:
                    file.write(generated_image.data)
                  print("Image saved to file: ", f"{chapter['name']}.png")
                  chapter_images.append(generated_image)
                  break
          break
  print("Chapter image interaction completed.")


print("Animating last chapter...")
# Pick the first chapter illustration to animate
chapter_index_to_animate = 0
first_chapter = chapters[chapter_index_to_animate]
print("Animating chapter: ", first_chapter['name'])

# Convert interaction image content to types.Image for Veo
image_bytes = base64.b64decode(chapter_images[chapter_index_to_animate].data)
veo_image = types.Image(image_bytes=image_bytes, mime_type=chapter_images[chapter_index_to_animate].mime_type)

# Generate a video from the image
operation = client.models.generate_videos(
    model=VEO_MODEL_ID,
    prompt=first_chapter['prompt'],
    image=veo_image,
    config=types.GenerateVideosConfig(
        aspect_ratio="16:9",
        resolution="720p",
    ),
)

# Wait for the video to be generated (takes about a minute)
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

for n, generated_video in enumerate(operation.result.generated_videos):
    client.files.download(file=generated_video.video)
    generated_video.video.save(f"chapter_video_{n}.mp4")
    print("Video saved to file: ", f"chapter_video_{n}.mp4")
