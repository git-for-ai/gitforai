# Docker Guide for GitForAI

## Quick Start

### 1. Set up environment variables (Optional)

**Default**: GitForAI uses local embeddings (free, privacy-preserving, offline).
No API keys or environment variables required for basic usage!

**Optional**: For OpenAI embeddings (~5-7% better quality):

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or your preferred editor
```

Optional environment variables for OpenAI provider:
- `EMBEDDING_PROVIDER=openai` - Use OpenAI instead of local embeddings
- `OPENAI_API_KEY` - Your OpenAI API key (required for openai provider)
- `ANTHROPIC_API_KEY` - Your Anthropic API key (for future features)

### 2. Build the Docker image

```bash
# Build production image with LLM support
docker-compose build

# Or build development image
docker-compose -f docker-compose.dev.yml build
```

## Usage Examples

### Basic Extraction (Phase 1)

```bash
# Extract commits from a repository
docker-compose run --rm gitforai extract /repos/myrepo --output /output/commits.json
```

### Semantic Enrichment (Phase 2)

**Default: Local embeddings (free, offline)**

```bash
# Enrich commits with local embeddings (default, no API key needed)
docker-compose run --rm gitforai enrich /repos/myrepo --output /output/enriched.json

# Limit to 50 commits
docker-compose run --rm gitforai enrich /repos/myrepo -n 50 --output /output/enriched.json

# With verbose output
docker-compose run --rm gitforai enrich /repos/myrepo -v --output /output/enriched.json

# Use different local model (faster but slightly lower quality)
docker-compose run --rm gitforai enrich /repos/myrepo --embedding-model paraphrase-MiniLM-L6-v2
```

**Optional: OpenAI embeddings (requires API key)**

```bash
# Use OpenAI provider for maximum quality
docker-compose run --rm gitforai enrich /repos/myrepo \
  --embedding-provider openai \
  --api-key $OPENAI_API_KEY \
  --output /output/enriched.json

# With custom OpenAI embedding model
docker-compose run --rm gitforai enrich /repos/myrepo \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large \
  --api-key $OPENAI_API_KEY
```

### Cache Management

```bash
# View cache statistics
docker-compose run --rm gitforai semantic-stats

# Clear the cache
docker-compose run --rm gitforai clear-cache --force
```

## Development Environment

### Interactive Development Shell

```bash
# Start development container
docker-compose -f docker-compose.dev.yml run --rm dev

# Inside container, you have full access to source code
gitforai --help
python -m pytest tests/
```

### Run Tests

```bash
# Run all tests
docker-compose -f docker-compose.dev.yml run --rm test

# Run specific test file
docker-compose -f docker-compose.dev.yml run --rm test pytest tests/unit/test_cache.py -v
```

### Code Linting

```bash
# Run linters and formatters
docker-compose -f docker-compose.dev.yml run --rm lint
```

### Interactive Python Shell

```bash
# Start Python shell with gitforai loaded
docker-compose -f docker-compose.dev.yml run --rm shell

# In the shell - Local provider (default):
# >>> from gitforai.llm import LocalProvider, SemanticProcessor
# >>> provider = LocalProvider(model="all-MiniLM-L6-v2")

# Or OpenAI provider:
# >>> from gitforai.llm import OpenAIProvider
# >>> provider = OpenAIProvider(api_key="your-key", model="gpt-4-turbo-preview")
```

## Volume Mounts

The Docker setup uses the following volumes:

- `/repos` - Mount your Git repositories here (read-only)
- `/output` - Output directory for extracted/enriched data
- `/cache` - LLM response cache (persistent across runs)
- `/data` - General data directory (Phase 3+)

## Example Docker Compose Override

Create `docker-compose.override.yml` for custom settings:

```yaml
version: '3.8'

services:
  gitforai:
    volumes:
      # Mount specific repositories
      - /path/to/your/repo:/repos/myrepo:ro
      - ./my-output:/output

    environment:
      # Override default model
      - LLM_MODEL=gpt-3.5-turbo

    # Default command
    command: enrich /repos/myrepo --output /output/enriched.json
```

Then run: `docker-compose up`

## Resource Configuration

The default resource limits are:

- **CPU**: 2 cores max, 0.5 cores reserved
- **Memory**: 2GB max, 512MB reserved

Adjust in `docker-compose.yml` under `deploy.resources` if needed.

## Troubleshooting

### API Key Not Found

**Note**: API keys are only required if you use `--embedding-provider openai`.
The default local embeddings work without any API keys.

If you see "OpenAI API key required" when using openai provider:
1. Ensure you've created a `.env` file from `.env.example`
2. Your `.env` file has valid `OPENAI_API_KEY`
3. Docker Compose is reading the `.env` file (it should be in the same directory)
4. Or pass the key directly: `--api-key $OPENAI_API_KEY`

### Model Download (First Run)

When using local embeddings for the first time, the model (~80-400MB) will be downloaded.
This is a one-time operation and takes 1-2 minutes. The model is cached for future runs.

### Cache Permission Issues

If you encounter permission errors with the cache:

```bash
# Reset cache volume
docker volume rm gitforai-cache
docker-compose up
```

### Image Size

The production image is optimized for size (~265MB base + dependencies):
- Uses Python 3.10-slim base
- Multi-stage build to exclude build tools
- Only includes runtime dependencies

To see image size:
```bash
docker images gitforai
```

## Security Notes

- API keys are passed via environment variables, never committed to Git
- Repositories are mounted read-only by default
- Container runs as non-root user `gitforai` (UID 1000)
- No sensitive data is stored in the image

## Next Steps

Once you're comfortable with Docker usage:
- See `README.md` for detailed feature documentation
- See `implementation-plan.md` for project roadmap
- Phase 3: Vector database integration (coming soon)
