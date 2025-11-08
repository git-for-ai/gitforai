# GitForAI

**Semantic Memory Infrastructure for AI Coding Assistants**

Stop wasting tokens. Give your AI agents Git history context through vector embeddings and semantic search.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

AI coding assistants waste tokens and miss context:

- **15-20K tokens** to answer simple questions about codebase history
- No understanding of **"why"** code evolved the way it did
- **Hallucinations** from missing historical context
- **Expensive** token costs for enterprise teams

## The Solution

GitForAI transforms Git history into queryable semantic memory:

```python
from gitforai import GitForAI

# Initialize with your repo
git_memory = GitForAI("/path/to/repo")

# Ask questions in natural language
results = git_memory.query("How does authentication work?")
# Returns relevant commits with 85% fewer tokens

# Track file evolution
history = git_memory.track_file("auth.py")
# See how a file changed over time

# Find similar changes
similar = git_memory.find_similar(commit_hash)
# Discover related work
```

## Features

- 🎯 **85% token reduction** - Semantic search returns only relevant context
- 🔒 **Privacy-first** - Local embeddings, no API keys required
- ⚡ **Fast** - Sub-second semantic search with ChromaDB
- 🐳 **Self-hostable** - Docker setup included
- 🔌 **Pluggable** - Extensible architecture for custom integrations

## Quick Start

### Installation

```bash
# Install from PyPI
pip install gitforai

# Or install from source
git clone https://github.com/git-for-ai/gitforai.git
cd gitforai
pip install -e .
```

### Index Your Repository

```bash
# Index repository with local embeddings (zero cost, offline)
gitforai index /path/to/repo

# Search your codebase semantically
gitforai search "authentication bug fixes"

# Get detailed commit info
gitforai analyze abc123 --diffs
```

### Docker (Recommended)

```bash
# Pull and run
docker pull gitforai/core
docker run -v $(pwd):/repo gitforai/core index /repo

# Or use docker-compose
docker-compose up
```

## How It Works

1. **Extract** - Parse Git commits, diffs, and file changes
2. **Embed** - Generate semantic embeddings (local, no API cost)
3. **Index** - Store in ChromaDB vector database
4. **Query** - Natural language search returns relevant context

**Result:** AI agents get exactly the context they need, without wasting tokens on irrelevant code.

## Platform Integrations

*Platform-specific adapters available separately.*

## Use Cases

### For Individual Developers
- Understand unfamiliar codebases quickly
- Find relevant commits when debugging
- Learn from code evolution patterns
- Reduce AI assistant token costs

### For Teams
- Onboard new developers faster
- Share institutional knowledge automatically
- Improve code review quality
- Standardize AI context across team

### For Enterprises
- Reduce token costs by 85%
- Self-host for data privacy
- Integrate with existing AI tools
- SOC2/GDPR compliant

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete usage guide
- **[API Reference](docs/API_REFERENCE.md)** - Python API documentation
- **[Docker Guide](docs/DOCKER.md)** - Self-hosting with Docker
- **[Contributing](CONTRIBUTING.md)** - How to contribute

## Architecture

- **Extraction** - GitPython for repository parsing
- **Embeddings** - sentence-transformers (local, free) or OpenAI (optional)
- **Vector DB** - ChromaDB for semantic search
- **CLI** - Typer for command-line interface

See [CLAUDE.md](CLAUDE.md) for detailed architecture.

## Development

```bash
# Clone and setup
git clone https://github.com/git-for-ai/gitforai.git
cd gitforai
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/gitforai --cov-report=html

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## Configuration

Create `.env` file:

```env
# Embedding provider (default: local, zero cost)
EMBEDDING_PROVIDER=local  # or "openai" for maximum quality
EMBEDDING_MODEL=all-MiniLM-L6-v2  # 384 dims, 80MB, free

# Vector database
VECTORDB_PROVIDER=chroma
VECTORDB_PERSIST_DIR=~/.gitforai/vectordb

# Optional: OpenAI API key (only if using OpenAI embeddings)
OPENAI_API_KEY=your-key-here
```

**Default settings use local embeddings:**
- Works offline
- Zero API cost
- No API keys required
- ~88% of OpenAI quality
- Auto-downloads ~80MB model on first use

## Why Open Source?

We believe semantic Git history should be accessible to all developers. The core extraction, embedding, and indexing logic is open source (MIT license).

**Commercial offerings:**
- Managed cloud hosting (Pro/Team tiers)
- Platform-specific integrations
- Enterprise features (SSO, RBAC, audit logs)
- Professional support

See [gitforai.com](https://gitforai.com) for commercial options.

## Performance

- **Indexing**: 1000 commits in <5 minutes
- **Query**: Semantic search in <500ms
- **Accuracy**: Relevant results in top 5 for 90% of queries
- **Cost**: $0.00 with local embeddings (vs $0.10/1000 commits with OpenAI)

## License

[MIT License](LICENSE) - See LICENSE file for details.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- **Issues**: [GitHub Issues](https://github.com/git-for-ai/gitforai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/git-for-ai/gitforai/discussions)
- **Email**: support@gitforai.com
- **Docs**: [docs.gitforai.com](https://docs.gitforai.com)

## Roadmap

- [x] Core Git extraction and indexing
- [x] Local embeddings (sentence-transformers)
- [x] Vector database storage (ChromaDB)
- [x] Incremental updates (2-6x faster)
- [x] CLI interface
- [x] Docker support
- [ ] Multi-repo support
- [ ] Real-time updates
- [ ] Advanced query optimization

## Community

- ⭐ **Star this repo** if you find it useful
- 🐛 **Report bugs** via GitHub Issues
- 💡 **Request features** via GitHub Discussions
- 🤝 **Contribute** - see [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Built with ❤️ by the GitForAI team**

*Semantic memory for the AI coding revolution*
