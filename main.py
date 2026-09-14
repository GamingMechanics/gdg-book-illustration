"""Generate character and chapter illustrations for a book, then animate the first chapter."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

IMAGE_MODEL_ID = "gemini-2.5-flash-image"  # Options are: "gemini-3.1-flash-lite-image", "gemini-2.5-flash-image", "gemini-3.1-flash-image" or "gemini-3-pro-image"
GEMINI_MODEL_ID = "gemini-3.5-flash" # Options are: "gemini-2.5-flash", "gemini-3.1-flash-lite-preview", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash" or "gemini-3.1-pro-preview"
VEO_MODEL_ID = "veo-3.1-lite-generate-preview" # Options are "veo-3.1-lite-generate-preview", "veo-3.1-fast-generate-preview" or "veo-3.1-generate-preview"

# Better models:
# IMAGE_MODEL_ID = "gemini-3.1-flash-image"  # Options are: "gemini-3.1-flash-lite-image", "gemini-2.5-flash-image", "gemini-3.1-flash-image" or "gemini-3-pro-image"
# GEMINI_MODEL_ID = "gemini-3.7-flash" # Options are: "gemini-2.5-flash", "gemini-3.1-flash-lite-preview", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash" or "gemini-3.1-pro-preview"
# VEO_MODEL_ID = "veo-3.1-generate-preview" # Options are "veo-3.1-lite-generate-preview", "veo-3.1-fast-generate-preview" or "veo-3.1-generate-preview"

SYSTEM_INSTRUCTIONS = """
  There must be no text on the image, it should not look like a cover page.
  It should be an full illustration with no borders, titles, nor description.
  Unless asked otherwise, stay family-friendly with uplifting colors.
  Each produced should be a simple image, no panels.
"""


class Prompt(BaseModel):
    name: str
    prompt: str


class BookInfo(BaseModel):
    title: str
    author: str

class Chapter(BaseModel):
    name: str
    prompt: str
    characters: list[str]

class Checkpoint(BaseModel):
    book_interaction_id: str | None = None
    style: str | None = None
    style_interaction_id: str | None = None
    characters_prompts_interaction_id: str | None = None
    last_image_interaction_id: str | None = None


BOOK_INFO_FILE = "book_info.json"
CHARACTERS_FILE = "characters.json"
CHAPTERS_FILE = "chapters.json"
CHECKPOINT_FILE = "checkpoint.json"


def prompt_response_format() -> dict[str, Any]:
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": {"type": "array", "items": Prompt.model_json_schema()},
    }


def book_info_response_format() -> dict[str, Any]:
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": BookInfo.model_json_schema(),
    }

def chapter_response_format() -> dict[str, Any]:
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": {"type": "array", "items": Chapter.model_json_schema()},
    }

@dataclass
class Settings:
    book_url: str = "https://www.gutenberg.org/cache/epub/113/pg113.txt" # default book from Project Gutenberg, The Secret Garden
    book_path: Path = field(default_factory=lambda: Path("data/book.txt"))
    output_dir: Path = field(default_factory=lambda: Path("data"))
    style: str = "graphic noir, dark graphic novels"
    service_tier: str = "standard" # "flex", "standard" or "priority"
    max_character_images: int = 5
    max_chapter_images: int = 3
    chapter_index_to_animate: int = 0
    animate_chapters: bool = True
    pause_after_checkpoint: bool = True
    late_ask_style: bool = False
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


def character_image_path(output_dir: Path, name: str) -> Path:
    return output_dir / f"{safe_filename(name)}.png"


def chapter_image_path(output_dir: Path, name: str) -> Path:
    return output_dir / f"{safe_filename(name)}.png"


def load_checkpoint(output_dir: Path) -> Checkpoint:
    path = output_dir / CHECKPOINT_FILE
    if not path.exists():
        return Checkpoint()
    return Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))


def save_checkpoint(output_dir: Path, checkpoint: Checkpoint) -> None:
    path = output_dir / CHECKPOINT_FILE
    path.write_text(
        json.dumps(checkpoint.model_dump(), indent=2),
        encoding="utf-8",
    )


def wait_for_continue(pause: bool) -> None:
    if pause:
        input("Press Enter to continue...")


def update_checkpoint(output_dir: Path, **fields: Any) -> Checkpoint:
    checkpoint = load_checkpoint(output_dir)
    updated = checkpoint.model_copy(update=fields)
    save_checkpoint(output_dir, updated)
    return updated


def load_prompts(path: Path) -> list[Prompt]:
    return [Prompt.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def load_chapters(path: Path) -> list[Chapter]:
    return [
        Chapter.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]


def save_prompts(path: Path, prompts: list[Prompt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([prompt.model_dump() for prompt in prompts], indent=2),
        encoding="utf-8",
    )
    logger.info("Prompts saved to %s", path)


def save_chapters(path: Path, chapters: list[Chapter]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([chapter.model_dump() for chapter in chapters], indent=2),
        encoding="utf-8",
    )
    logger.info("Chapters saved to %s", path)


def save_book_info(output_dir: Path, book_info: BookInfo) -> None:
    path = output_dir / BOOK_INFO_FILE
    path.write_text(book_info.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Book info saved to %s", path)


def load_book_info(output_dir: Path) -> BookInfo | None:
    path = output_dir / BOOK_INFO_FILE
    if not path.exists():
        return None
    return BookInfo.model_validate_json(path.read_text(encoding="utf-8"))


def all_character_images_exist(
    output_dir: Path, characters: list[Prompt], max_images: int
) -> bool:
    return all(
        character_image_path(output_dir, character.name).exists()
        for character in characters[:max_images]
    )


def all_chapter_images_exist(output_dir: Path, chapters: list[Chapter]) -> bool:
    return all(
        chapter_image_path(output_dir, chapter.name).exists() for chapter in chapters
    )


def load_saved_image(path: Path) -> types.Image:
    return types.Image(image_bytes=path.read_bytes(), mime_type="image/png")


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

def get_character_reference_images(
    characters: list[Prompt],
    output_dir: Path,
    requested_character_names: list[str],
) -> list[dict[str, str]]:
    """Return interaction image inputs for the requested characters."""
    characters_by_name = {character.name: character for character in characters}
    image_inputs: list[dict[str, str]] = []

    for character_name in requested_character_names:
        character = characters_by_name.get(character_name)
        if character is None:
            logger.warning("Character %r not found in saved character prompts", character_name)
            continue

        image_path = character_image_path(output_dir, character.name)
        if not image_path.exists():
            logger.warning("No image available for %r at %s", character_name, image_path)
            continue

        image_inputs.append(
            {
                "type": "image",
                "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                "mime_type": "image/png",
            }
        )

    return image_inputs


def create_book_interaction(
    client: genai.Client, book_uri: str, service_tier: str
) -> tuple[BookInfo, Any]:
    logger.info("Creating initial book interaction")
    interaction = client.interactions.create(
        model=GEMINI_MODEL_ID,
        input=[
            {
                "type": "text",
                "text": (
                    "Here's a book, to illustrate using Nano Banana. "
                    "Extract only the book's title and author from the document. "
                    "Do not add any other commentary."
                ),
            },
            {"type": "document", "uri": book_uri},
        ],
        response_format=book_info_response_format(),
        service_tier=service_tier,
    )
    raw_text = interaction.output_text
    if not raw_text and interaction.steps:
        raw_text = interaction.steps[-1].content[0].text

    book_info = BookInfo.model_validate(json.loads(raw_text))
    logger.info("Book identified: %s by %s", book_info.title, book_info.author)
    return book_info, interaction


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
                "a twist? Just give us the prompt for the art style that will "
                "added to the future prompts."
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

def parse_chapters_json(interaction: Any) -> list[Chapter]:
    raw_text = interaction.output_text
    if not raw_text and interaction.steps:
        raw_text = interaction.steps[-1].content[0].text

    chapters = json.loads(raw_text)
    return [Chapter.model_validate(item) for item in chapters]

def generate_character_prompts(
    client: genai.Client,
    style_interaction_id: str,
    service_tier: str,
    output_dir: Path,
    pause_after_checkpoint: bool = True,
) -> tuple[list[Prompt], str]:
    characters_path = output_dir / CHARACTERS_FILE
    if characters_path.exists():
        checkpoint = load_checkpoint(output_dir)
        if not checkpoint.characters_prompts_interaction_id:
            raise ValueError(
                f"Found saved character prompts at {characters_path}, but "
                f"{CHECKPOINT_FILE} is missing characters_prompts_interaction_id. "
                "Delete the saved files to start fresh."
            )
        characters = load_prompts(characters_path)
        logger.info("Loaded character prompts from %s", characters_path)
        return characters, checkpoint.characters_prompts_interaction_id

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
    save_prompts(characters_path, characters)
    update_checkpoint(
        output_dir, characters_prompts_interaction_id=interaction.id
    )
    logger.info(
        "Characters:\n%s",
        json.dumps([character.model_dump() for character in characters], indent=4),
    )
    wait_for_continue(pause_after_checkpoint)
    return characters, interaction.id


def extract_image_from_interaction(interaction: Any) -> Any | None:
    output_image = getattr(interaction, "output_image", None)
    if output_image:
        return output_image

    for step in reversed(interaction.steps):
        if step.type != "model_output" or not step.content:
            continue
        for content in reversed(step.content):
            if content.type == "image":
                return content
    return None


def save_interaction_image(image_content: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(image_content.data))
    logger.info("Image saved to %s", path)


def generate_character_images(
    client: genai.Client,
    characters: list[Prompt],
    style: str,
    book_info: BookInfo,
    output_dir: Path,
    max_images: int,
    service_tier: str,
    start_interaction_id: str | None = None,
    pause_after_checkpoint: bool = True,
) -> str:
    if all_character_images_exist(output_dir, characters, max_images):
        checkpoint = load_checkpoint(output_dir)
        if not checkpoint.last_image_interaction_id:
            raise ValueError(
                "Character images exist on disk, but checkpoint.json is missing "
                "last_image_interaction_id. Delete the images or checkpoint to "
                "start fresh."
            )
        logger.info("All character images already exist, skipping generation")
        return checkpoint.last_image_interaction_id

    logger.info("Generating character images")
    interaction_id = start_interaction_id

    if interaction_id is None:
        interaction = client.interactions.create(
            model=IMAGE_MODEL_ID,
            input=(
                "You are going to generate portrait images to illustrate "
                f"{book_info.title} from {book_info.author}. "
                f"The style we want you to follow is: {style} "
                f"Also follow those rules: {SYSTEM_INSTRUCTIONS}"
            ),
            service_tier=service_tier,
        )
        interaction_id = interaction.id
        update_checkpoint(output_dir, last_image_interaction_id=interaction_id)

    for character in characters[:max_images]:
        image_path = character_image_path(output_dir, character.name)
        if image_path.exists():
            logger.info("Character image already exists: %s", image_path)
            continue

        logger.info("Creating image for character: %s", character.name)
        interaction = client.interactions.create(
            model=IMAGE_MODEL_ID,
            input=(
                f"Create an illustration for {character.name} following this "
                f"description: {character.prompt}"
            ),
            previous_interaction_id=interaction_id,
            service_tier=service_tier,
        )
        interaction_id = interaction.id

        image = extract_image_from_interaction(interaction)
        if image:
            save_interaction_image(image, image_path)
        else:
            logger.warning("No image generated for %s", character.name)

        update_checkpoint(output_dir, last_image_interaction_id=interaction_id)

    logger.info("Character image generation completed")
    wait_for_continue(pause_after_checkpoint)
    return interaction_id


def generate_chapter_prompts(
    client: genai.Client,
    characters_interaction_id: str,
    service_tier: str,
    max_chapters: int,
    output_dir: Path,
) -> list[Chapter]:
    chapters_path = output_dir / CHAPTERS_FILE
    if chapters_path.exists():
        chapters = load_chapters(chapters_path)[:max_chapters]
        logger.info("Loaded chapter prompts from %s", chapters_path)
        return chapters

    logger.info("Generating chapter prompts")
    interaction = client.interactions.create(
        model=GEMINI_MODEL_ID,
        input=(
            "Now, for each chapters of the book, give me a prompt to illustrate "
            "what happens in it. It should be a single image, not a multi-tiled "
            "page. Be very descriptive, especially of the characters. Be very "
            "descriptive and remember to tell their name and to reuse the character "
            "prompts if they appear in the images. Also list all characters who "
            "appear in it. Each prompt should be at least 100 words but no more than 200 words."
        ),
        previous_interaction_id=characters_interaction_id,
        response_format=chapter_response_format(),
        service_tier=service_tier,
    )
    chapters = parse_chapters_json(interaction)[:max_chapters]
    save_chapters(chapters_path, chapters)
    logger.info(
        "Chapters:\n%s",
        json.dumps([chapter.model_dump() for chapter in chapters], indent=4),
    )
    return chapters


def generate_chapter_images(
    client: genai.Client,
    chapters: list[Chapter],
    characters: list[Prompt],
    output_dir: Path,
    service_tier: str,
    pause_after_checkpoint: bool,
) -> None:
    if all_chapter_images_exist(output_dir, chapters):
        logger.info("All chapter images already exist, skipping generation")
        return

    logger.info("Generating chapter images")

    for chapter in chapters:
        image_path = chapter_image_path(output_dir, chapter.name)
        if image_path.exists():
            logger.info("Chapter image already exists: %s", image_path)
            continue

        logger.info("Creating image for chapter: %s", chapter.name)
        image_inputs = get_character_reference_images(
            characters, output_dir, chapter.characters
        )
        interaction = client.interactions.create(
            model=IMAGE_MODEL_ID,
            input=[
                {
                    "type": "text",
                    "text": (
                        f"Create this illustration for {chapter.name}:\n"
                        f"{chapter.prompt}\n"
                        "Use the provided images as references of what the "
                        "characters look like."
                    ),
                },
                *image_inputs,
            ],
            system_instruction=SYSTEM_INSTRUCTIONS,
            service_tier=service_tier,
        )

        image = extract_image_from_interaction(interaction)
        if image:
            save_interaction_image(image, image_path)
        else:
            logger.warning("No image generated for %s", chapter.name)

    logger.info("Chapter image generation completed")
    wait_for_continue(pause_after_checkpoint)


def chapter_video_paths(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("chapter_video_*.mp4"))


def animate_chapter(
    client: genai.Client,
    chapter: Chapter,
    chapter_image: Path,
    output_dir: Path,
    poll_interval_seconds: int = 20,
) -> list[Path]:
    existing_videos = chapter_video_paths(output_dir)
    if existing_videos:
        logger.info("Chapter video already exists, skipping animation")
        return existing_videos

    logger.info("Animating chapter: %s", chapter.name)

    veo_image = load_saved_image(chapter_image)

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

    if operation.error:
        raise RuntimeError(f"Video generation failed: {operation.error}")

    result = operation.result or operation.response
    generated_videos = result.generated_videos if result else None
    if not generated_videos:
        reasons = []
        if result is not None:
            reasons = result.rai_media_filtered_reasons or []
            if result.rai_media_filtered_count:
                reasons = reasons or [
                    f"{result.rai_media_filtered_count} video(s) filtered by RAI"
                ]
        detail = "; ".join(reasons) if reasons else "no videos returned"
        raise RuntimeError(
            "Video generation completed but returned no videos "
            f"({detail}). Veo often filters scenes that depict children "
            "or other restricted people; try a chapter/image without minors, "
            "or a book with adult protagonists."
        )

    saved_paths = []
    for index, generated_video in enumerate(generated_videos):
        client.files.download(file=generated_video.video)
        video_path = output_dir / f"chapter_video_{index}.mp4"
        generated_video.video.save(str(video_path))
        logger.info("Video saved to %s", video_path)
        saved_paths.append(video_path)

    return saved_paths


def needs_api_setup(output_dir: Path, settings: Settings) -> bool:
    checkpoint = load_checkpoint(output_dir)
    characters_path = output_dir / CHARACTERS_FILE
    chapters_path = output_dir / CHAPTERS_FILE

    if load_book_info(output_dir) is None:
        return True
    if not checkpoint.style_interaction_id:
        return True
    if not characters_path.exists():
        return True
    if not all_character_images_exist(
        output_dir, load_prompts(characters_path), settings.max_character_images
    ):
        return not checkpoint.last_image_interaction_id
    if not chapters_path.exists():
        return not checkpoint.characters_prompts_interaction_id
    if not all_chapter_images_exist(
        output_dir, load_chapters(chapters_path)[: settings.max_chapter_images]
    ):
        return False
    return False


def run_pipeline(settings: Settings) -> None:
    client = create_client(settings)
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(output_dir)

    book_info = load_book_info(output_dir)

    if needs_api_setup(output_dir, settings):
        if not settings.book_path.exists():
            download_book(settings.book_url, settings.book_path)

        if book_info is None or not checkpoint.book_interaction_id:
            uploaded_book = upload_book(client, settings.book_path)
            book_info, book_interaction = create_book_interaction(
                client, uploaded_book.uri, settings.service_tier
            )
            save_book_info(output_dir, book_info)
            update_checkpoint(output_dir, book_interaction_id=book_interaction.id)
            wait_for_continue(settings.pause_after_checkpoint)
            book_interaction_id = book_interaction.id
        else:
            book_interaction_id = checkpoint.book_interaction_id

        if settings.late_ask_style and not (
            checkpoint.style and checkpoint.style_interaction_id
        ):
            settings = replace(settings, style=prompt_art_style(settings.style))

        if checkpoint.style and checkpoint.style_interaction_id:
            style = checkpoint.style
            style_interaction_id = checkpoint.style_interaction_id
            logger.info("Using saved art style from checkpoint")
        else:
            style, style_interaction_id = define_art_style(
                client,
                book_interaction_id,
                settings.style,
                settings.service_tier,
            )
            update_checkpoint(
                output_dir,
                style=style,
                style_interaction_id=style_interaction_id,
            )
            wait_for_continue(settings.pause_after_checkpoint)
    else:
        book_info = load_book_info(output_dir)
        if book_info is None:
            raise ValueError(f"Missing {BOOK_INFO_FILE} in {output_dir}")
        style = checkpoint.style or f'Follow this style: "{settings.style}" '
        style_interaction_id = checkpoint.style_interaction_id or ""

    characters, characters_prompts_interaction_id = generate_character_prompts(
        client,
        style_interaction_id,
        settings.service_tier,
        output_dir,
        pause_after_checkpoint=settings.pause_after_checkpoint,
    )

    checkpoint = load_checkpoint(output_dir)
    generate_character_images(
        client,
        characters,
        style,
        book_info,
        output_dir,
        settings.max_character_images,
        settings.service_tier,
        start_interaction_id=checkpoint.last_image_interaction_id,
        pause_after_checkpoint=settings.pause_after_checkpoint,
    )

    chapters = generate_chapter_prompts(
        client,
        characters_prompts_interaction_id,
        settings.service_tier,
        settings.max_chapter_images,
        output_dir,
    )

    generate_chapter_images(
        client,
        chapters,
        characters,
        output_dir,
        settings.service_tier,
        pause_after_checkpoint=settings.pause_after_checkpoint,
    )

    if not settings.animate_chapters:
        return

    index = settings.chapter_index_to_animate
    if index >= len(chapters):
        raise IndexError(
            f"chapter_index_to_animate={index} is out of range for "
            f"{len(chapters)} chapters"
        )

    chapter_image = chapter_image_path(output_dir, chapters[index].name)
    if not chapter_image.exists():
        raise FileNotFoundError(
            f"Cannot animate chapter {chapters[index].name!r}: "
            f"missing image at {chapter_image}"
        )

    animate_chapter(
        client,
        chapters[index],
        chapter_image,
        output_dir,
    )


def prompt_art_style(default: str) -> str:
    style_input = input(f"Art style [{default}]: ").strip()
    return style_input or default


def prompt_settings(
    defaults: Settings | None = None, *, ask_style: bool = True
) -> Settings:
    settings = defaults or Settings()

    book_url_input = input(
        f"Book URL [{settings.book_url}]: "
    ).strip()
    style = prompt_art_style(settings.style) if ask_style else settings.style

    return replace(
        settings,
        book_url=book_url_input or settings.book_url,
        style=style,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate character and chapter illustrations for a book."
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not wait for Enter after each checkpoint update.",
    )
    parser.add_argument(
        "--late-ask-style",
        action="store_true",
        help="Ask for the art style after book identification instead of at startup.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    settings = prompt_settings(
        Settings(
            pause_after_checkpoint=not args.no_pause,
            late_ask_style=args.late_ask_style,
        ),
        ask_style=not args.late_ask_style,
    )
    run_pipeline(settings)


if __name__ == "__main__":
    main()
