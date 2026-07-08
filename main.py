"""Generate character and chapter illustrations for a book, then animate the first chapter."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

IMAGE_MODEL_ID = "gemini-2.5-flash-image"
GEMINI_MODEL_ID = "gemini-3.5-flash"
VEO_MODEL_ID = "veo-3.1-lite-generate-preview"

SYSTEM_INSTRUCTIONS = """
  There must be no text on the image, it should not look like a cover page.
  It should be an full illustration with no borders, titles, nor description.
  Unless asked otherwise, stay family-friendly with uplifting colors.
  Each produced should be a simple image, no panels.
"""


class Prompt(BaseModel):
    name: str
    prompt: str


def prompt_response_format() -> dict[str, Any]:
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": {"type": "array", "items": Prompt.model_json_schema()},
    }


@dataclass
class Settings:
    book_url: str = "https://www.gutenberg.org/cache/epub/730/pg730.txt"
    book_path: Path = field(default_factory=lambda: Path("book.txt"))
    output_dir: Path = field(default_factory=Path.cwd)
    style: str = "comic book"
    service_tier: str = "flex"
    max_character_images: int = 5
    max_chapter_images: int = 3
    chapter_index_to_animate: int = 0
    animate_chapters: bool = True
    google_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )


def create_client(settings: Settings) -> genai.Client:
    if not settings.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to a .env file or export it "
            "in your environment before running this script."
        )

    return genai.Client(
        api_key=settings.google_api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=5,
                initial_delay=2.0,
                max_delay=60.0,
                http_status_codes=[429, 500, 502, 503, 504],
            )
        ),
    )


def safe_filename(name: str) -> str:
    sanitised = re.sub(r"[^\w\-. ]", "_", name).strip()
    return sanitised or "unnamed"


def download_book(url: str, destination: Path) -> Path:
    logger.info("Downloading book from %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    logger.info("Book saved to %s", destination)
    return destination


def upload_book(client: genai.Client, book_path: Path) -> Any:
    logger.info("Uploading book to Gemini")
    uploaded = client.files.upload(file=str(book_path))
    logger.info("Book uploaded")
    return uploaded


def create_book_interaction(
    client: genai.Client, book_uri: str, service_tier: str
) -> Any:
    logger.info("Creating initial book interaction")
    return client.interactions.create(
        model=GEMINI_MODEL_ID,
        input=[
            {
                "type": "text",
                "text": (
                    "Here's a book, to illustrate using Nano Banana. "
                    "Don't say anything for now, instructions will follow."
                ),
            },
            {"type": "document", "uri": book_uri},
        ],
        service_tier=service_tier,
    )


def define_art_style(
    client: genai.Client,
    book_interaction_id: str,
    style: str,
    service_tier: str,
) -> tuple[str, str]:
    """Return the formatted style prompt and the style interaction id."""
    logger.info("Defining art style")

    if style:
        interaction = client.interactions.create(
            model=GEMINI_MODEL_ID,
            input=(
                f'The art style will be:"{style}". Keep that in mind when '
                "generating future prompts. Keep quiet for now, instructions "
                "will follow."
            ),
            previous_interaction_id=book_interaction_id,
            service_tier=service_tier,
        )
        resolved_style = style
    else:
        interaction = client.interactions.create(
            model=GEMINI_MODEL_ID,
            input=(
                "Can you define a art style that would fit the story but with "
                "a twist? Just give us the prompt for the art syle that will "
                "added to the furture prompts."
            ),
            previous_interaction_id=book_interaction_id,
            service_tier=service_tier,
        )
        resolved_style = interaction.output_text

    formatted_style = f'Follow this style: "{resolved_style}" '
    logger.info("Art style defined: %s", formatted_style)
    return formatted_style, interaction.id


def parse_prompts_json(interaction: Any) -> list[Prompt]:
    raw_text = interaction.output_text
    if not raw_text and interaction.steps:
        raw_text = interaction.steps[-1].content[0].text

    prompts = json.loads(raw_text)
    return [Prompt.model_validate(item) for item in prompts]


def generate_character_prompts(
    client: genai.Client,
    style_interaction_id: str,
    service_tier: str,
) -> tuple[list[Prompt], str]:
    logger.info("Generating character prompts")
    interaction = client.interactions.create(
        model=GEMINI_MODEL_ID,
        input=(
            "Can you describe the main characters and prepare a prompt describing "
            "them with as much details as possible (use the descriptions from the "
            "book) so Nano Banana can generate images of them? Each prompt should "
            "be at least 50 words."
        ),
        previous_interaction_id=style_interaction_id,
        response_format=prompt_response_format(),
        service_tier=service_tier,
    )
    characters = parse_prompts_json(interaction)
    logger.info(
        "Characters:\n%s",
        json.dumps([character.model_dump() for character in characters], indent=4),
    )
    return characters, interaction.id


def extract_image_from_interaction(interaction: Any) -> Any | None:
    for step in reversed(interaction.steps):
        if step.type != "model_output" or not step.content:
            continue
        for content in reversed(step.content):
            if content.type == "image":
                return content
    return None


def save_interaction_image(image_content: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_content.data)
    logger.info("Image saved to %s", path)


def generate_character_images(
    client: genai.Client,
    characters: list[Prompt],
    style: str,
    output_dir: Path,
    max_images: int,
    service_tier: str,
) -> tuple[list[Any | None], Any]:
    logger.info("Generating character images")
    interaction = client.interactions.create(
        model=IMAGE_MODEL_ID,
        input=(
            "You are going to generate portrait images to illustrate "
            "The Wind in the Willows from Kenneth Grahame. "
            f"The style we want you to follow is: {style} "
            f"Also follow those rules: {SYSTEM_INSTRUCTIONS}"
        ),
        service_tier=service_tier,
    )

    generated_images: list[Any | None] = []

    for character in characters[:max_images]:
        logger.info("Creating image for character: %s", character.name)
        interaction = client.interactions.create(
            model=IMAGE_MODEL_ID,
            input=(
                f"Create an illustration for {character.name} following this "
                f"description: {character.prompt}"
            ),
            previous_interaction_id=interaction.id,
            service_tier=service_tier,
        )

        image = extract_image_from_interaction(interaction)
        if image:
            save_interaction_image(
                image, output_dir / f"{safe_filename(character.name)}.png"
            )
            generated_images.append(image)
        else:
            logger.warning("No image generated for %s", character.name)
            generated_images.append(None)

    logger.info("Character image generation completed")
    return generated_images, interaction


def generate_chapter_prompts(
    client: genai.Client,
    characters_interaction_id: str,
    service_tier: str,
    max_chapters: int,
) -> list[Prompt]:
    logger.info("Generating chapter prompts")
    interaction = client.interactions.create(
        model=GEMINI_MODEL_ID,
        input=(
            "Now, for each chapters of the book, give me a prompt to illustrate "
            "what happens in it. It should be a single image, not a multi-tiled "
            "page. Be very descriptive, especially of the characters. Be very "
            "descriptive and remember to tell their name and to reuse the character "
            "prompts if they appear in the images. Also list all characters who "
            "appear in it."
        ),
        previous_interaction_id=characters_interaction_id,
        response_format=prompt_response_format(),
        service_tier=service_tier,
    )
    chapters = parse_prompts_json(interaction)[:max_chapters]
    logger.info(
        "Chapters:\n%s",
        json.dumps([chapter.model_dump() for chapter in chapters], indent=4),
    )
    return chapters


def generate_chapter_images(
    client: genai.Client,
    chapters: list[Prompt],
    previous_image_interaction_id: str,
    output_dir: Path,
    service_tier: str,
) -> tuple[list[Any], Any]:
    logger.info("Generating chapter images")
    interaction = client.interactions.create(
        model=IMAGE_MODEL_ID,
        input=(
            "Starting from now, we're going to illustrate the book's chapters. "
            "Don't forget to refer to your previous illustrations of the characters "
            "to keep the characters consistency, but feel free to change their "
            "position."
        ),
        previous_interaction_id=previous_image_interaction_id,
        service_tier=service_tier,
    )

    chapter_images: list[Any] = []

    for chapter in chapters:
        logger.info("Creating image for chapter: %s", chapter.name)
        interaction = client.interactions.create(
            model=IMAGE_MODEL_ID,
            input=(
                f"Create an illustration for {chapter.name} using the previously "
                f"generated characters following this description: {chapter.prompt}"
            ),
            previous_interaction_id=interaction.id,
            service_tier=service_tier,
        )

        image = extract_image_from_interaction(interaction)
        if image:
            save_interaction_image(
                image, output_dir / f"{safe_filename(chapter.name)}.png"
            )
            chapter_images.append(image)
        else:
            logger.warning("No image generated for %s", chapter.name)

    logger.info("Chapter image generation completed")
    return chapter_images, interaction


def animate_chapter(
    client: genai.Client,
    chapter: Prompt,
    chapter_image: Any,
    output_dir: Path,
    poll_interval_seconds: int = 20,
) -> list[Path]:
    logger.info("Animating chapter: %s", chapter.name)

    image_bytes = base64.b64decode(chapter_image.data)
    veo_image = types.Image(
        image_bytes=image_bytes,
        mime_type=chapter_image.mime_type,
    )

    operation = client.models.generate_videos(
        model=VEO_MODEL_ID,
        prompt=chapter.prompt,
        image=veo_image,
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            resolution="720p",
        ),
    )

    while not operation.done:
        time.sleep(poll_interval_seconds)
        operation = client.operations.get(operation)

    saved_paths: list[Path] = []
    for index, generated_video in enumerate(operation.result.generated_videos):
        client.files.download(file=generated_video.video)
        video_path = output_dir / f"chapter_video_{index}.mp4"
        generated_video.video.save(str(video_path))
        logger.info("Video saved to %s", video_path)
        saved_paths.append(video_path)

    return saved_paths


def run_pipeline(settings: Settings) -> None:
    client = create_client(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    download_book(settings.book_url, settings.book_path)
    uploaded_book = upload_book(client, settings.book_path)

    book_interaction = create_book_interaction(
        client, uploaded_book.uri, settings.service_tier
    )

    style, style_interaction_id = define_art_style(
        client, book_interaction.id, settings.style, settings.service_tier
    )

    characters, characters_prompts_interaction_id = generate_character_prompts(
        client, style_interaction_id, settings.service_tier
    )

    _, last_image_interaction = generate_character_images(
        client,
        characters,
        style,
        settings.output_dir,
        settings.max_character_images,
        settings.service_tier,
    )

    chapters = generate_chapter_prompts(
        client,
        characters_prompts_interaction_id,
        settings.service_tier,
        settings.max_chapter_images,
    )

    chapter_images, _ = generate_chapter_images(
        client,
        chapters,
        last_image_interaction.id,
        settings.output_dir,
        settings.service_tier,
    )

    if not settings.animate_chapters or not chapter_images:
        return

    index = settings.chapter_index_to_animate
    if index >= len(chapters) or index >= len(chapter_images):
        raise IndexError(
            f"chapter_index_to_animate={index} is out of range for "
            f"{len(chapters)} chapters"
        )

    animate_chapter(
        client,
        chapters[index],
        chapter_images[index],
        settings.output_dir,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline(Settings())


if __name__ == "__main__":
    main()
