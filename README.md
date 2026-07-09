# Book Illustration

Generate AI illustrations for a book using Google Gemini: character portraits, chapter scenes, and an animated video of the first chapter.

The pipeline downloads a plain-text book (default: [_The Secret Garden_](https://www.gutenberg.org/cache/epub/113/pg113.txt) from Project Gutenberg), analyses it with Gemini, defines an art style, and produces images with Gemini’s image model. It then animates one chapter illustration with Veo.

## What it does

When you run the script, it performs these steps in order:

1. **Download the book** — Fetches the text from a URL (or uses an existing local file).
2. **Identify the book** — Uploads the text to Gemini and extracts the title and author.
3. **Define an art style** — Uses your chosen style (or asks Gemini to suggest one).
4. **Generate character prompts** — Describes the main characters in detail for image generation.
5. **Generate character images** — Creates portrait illustrations for up to five characters.
6. **Generate chapter prompts** — Writes illustration prompts for up to three chapters.
7. **Generate chapter images** — Illustrates each chapter, using character images as visual references.
8. **Animate a chapter** — Turns the first chapter illustration into a short video with Veo.

Progress is saved under `data/`, so you can stop and resume a run without repeating completed steps. See [Output files](#output-files) below.

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or another way to install dependencies from `pyproject.toml`
- A **Google AI API key** with access to:
  - Gemini text models (e.g. `gemini-3.5-flash`)
  - Gemini image generation (e.g. `gemini-2.5-flash-image`)
  - Veo video generation (e.g. `veo-3.1-lite-generate-preview`)

Create a key in [Google AI Studio](https://aistudio.google.com/apikey).

## Setup

1. **Clone or download this repository**, then open a terminal in the project directory.

2. **Install dependencies** with uv:

   ```bash
   uv sync
   ```

   This creates a virtual environment and installs the packages listed in `pyproject.toml`.

3. **Configure your API key** by creating a `.env` file in the project root:

   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

   The script loads this automatically via `python-dotenv`. Do not commit `.env` to version control.

## Running

Start the pipeline:

```bash
uv run python main.py
```

You will be prompted for:

- **Book URL** — Press Enter to use the default Project Gutenberg link, or paste another plain-text book URL.
- **Art style** — Press Enter for the default (`graphic noir, dark graphic novels`), or describe your preferred style.

The script logs each step to the terminal. Image and video generation can take several minutes and uses paid API quota.

### Customising behaviour

Most options live in the `Settings` dataclass at the top of `main.py`. You can change defaults there or call `run_pipeline()` from your own script:

| Setting | Default | Description |
|--------|---------|-------------|
| `book_url` | Project Gutenberg URL for _The Secret Garden_ | Source URL for the book text |
| `book_path` | `data/book.txt` | Where the downloaded text is saved |
| `output_dir` | `data` | Directory for all generated files |
| `style` | `graphic noir, dark graphic novels` | Art style passed to the model |
| `max_character_images` | `5` | Maximum character portraits to generate |
| `max_chapter_images` | `3` | Maximum chapter illustrations to generate |
| `chapter_index_to_animate` | `0` | Which chapter (0-based) to animate |
| `animate_chapters` | `true` | Set to `false` to skip video generation |
| `service_tier` | `standard` | API tier: `flex`, `standard`, or `priority` |

Model IDs (`IMAGE_MODEL_ID`, `GEMINI_MODEL_ID`, `VEO_MODEL_ID`) are also defined near the top of `main.py`.

## Output files

All artefacts are written to `output_dir` (default: `data/`). That directory is gitignored.

| File | Description |
|------|-------------|
| `book.txt` | Downloaded book text |
| `book_info.json` | Title and author extracted by Gemini |
| `checkpoint.json` | Interaction IDs and style for resuming |
| `characters.json` | Character names and image prompts |
| `chapters.json` | Chapter names, prompts, and character lists |
| `<Character Name>.png` | Character portrait images |
| `<Chapter Name>.png` | Chapter illustration images |
| `chapter_video_0.mp4` | Animated video of the selected chapter |

To start fresh for a book, delete its output folder (or the relevant JSON/PNG/MP4 files). If you remove only some files, ensure `checkpoint.json` stays consistent or delete it as well.

## Models used

- **Text & prompts:** `gemini-3.5-flash` (configurable via `GEMINI_MODEL_ID`)
- **Images:** `gemini-2.5-flash-image` (configurable via `IMAGE_MODEL_ID`)
- **Video:** `veo-3.1-lite-generate-preview` (configurable via `VEO_MODEL_ID`)

Alternative model names are listed in comments next to each constant in `main.py`.

## Troubleshooting

- **`GOOGLE_API_KEY is not set`** — Add the key to `.env` or export it in your shell before running.
- **Rate limits or server errors** — The client retries automatically; wait and run again. Completed steps are skipped thanks to checkpoints.
- **Inconsistent checkpoint state** — Delete `checkpoint.json` and any partial outputs, then re-run.
