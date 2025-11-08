"""Command-line interface for GitForAI."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from gitforai.extraction import GitExtractor
from gitforai.models import RepositoryConfig

app = typer.Typer(
    name="gitforai",
    help="Git History Mining for AI Agents - Extract and analyze Git repository history",
    add_completion=False,
)
console = Console()


@app.command()
def extract(
    repo_path: Path = typer.Argument(..., help="Path to Git repository"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    max_commits: Optional[int] = typer.Option(None, "--max-commits", "-n", help="Maximum commits to extract"),
    branch: str = typer.Option("HEAD", "--branch", "-b", help="Branch to extract from"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Extract commit metadata from a Git repository."""
    try:
        # Create repository config
        config = RepositoryConfig(repo_path=repo_path)
        extractor = GitExtractor(config)

        console.print(f"[bold green]Extracting commits from:[/bold green] {repo_path}")
        console.print(f"[bold blue]Branch:[/bold blue] {branch}")

        commits = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting commits...", total=None)

            for commit_meta in extractor.extract_all_commits(branch=branch, max_count=max_commits):
                commits.append(commit_meta.model_dump(mode="json"))

                if verbose:
                    console.print(
                        f"  [cyan]{commit_meta.short_hash}[/cyan] "
                        f"{commit_meta.message_summary} "
                        f"[dim]by {commit_meta.author_name}[/dim]"
                    )

            progress.update(task, completed=True)

        console.print(f"\n[bold green]✓[/bold green] Extracted {len(commits)} commits")

        # Save to file if output specified
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(commits, f, indent=2, default=str)
            console.print(f"[bold green]✓[/bold green] Saved to {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def analyze(
    repo_path: Path = typer.Argument(..., help="Path to Git repository"),
    commit_hash: str = typer.Argument(..., help="Commit hash to analyze"),
    show_diffs: bool = typer.Option(False, "--diffs", "-d", help="Show file diffs"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Show file contents"),
) -> None:
    """Analyze a specific commit in detail."""
    try:
        config = RepositoryConfig(repo_path=repo_path)
        extractor = GitExtractor(config)

        # Extract commit metadata
        commit = extractor.extract_commit(commit_hash)

        # Display commit info
        console.print("\n[bold]Commit Information[/bold]")
        console.print(f"[cyan]Hash:[/cyan] {commit.hash}")
        console.print(f"[cyan]Author:[/cyan] {commit.author_name} <{commit.author_email}>")
        console.print(f"[cyan]Date:[/cyan] {commit.timestamp}")
        console.print(f"[cyan]Message:[/cyan] {commit.message}")
        console.print(f"[cyan]Files Changed:[/cyan] {len(commit.files_changed)}")
        console.print(f"[cyan]Is Merge:[/cyan] {commit.is_merge}")

        if commit.files_changed:
            console.print("\n[bold]Changed Files:[/bold]")
            for file_path in commit.files_changed:
                console.print(f"  • {file_path}")

        # Extract and show diffs
        if show_diffs:
            console.print("\n[bold]File Diffs:[/bold]")
            diffs = extractor.extract_commit_diffs(commit_hash)

            for diff in diffs:
                console.print(f"\n[yellow]{diff.file_path}[/yellow] ({diff.change_type})")
                console.print(f"  +{diff.additions} -{diff.deletions}")

                if diff.diff_text and not diff.is_binary:
                    # Show first 20 lines of diff
                    lines = diff.diff_text.split("\n")[:20]
                    console.print("  " + "\n  ".join(lines))
                    if len(diff.diff_text.split("\n")) > 20:
                        console.print("  [dim]... (truncated)[/dim]")

        # Extract and show file contents
        if show_content:
            console.print("\n[bold]File Contents:[/bold]")
            snapshots = extractor.extract_all_snapshots(commit_hash)

            for snapshot in snapshots[:5]:  # Limit to first 5 files
                console.print(f"\n[yellow]{snapshot.file_path}[/yellow] ({snapshot.size_bytes} bytes)")
                if not snapshot.is_binary:
                    lines = snapshot.content.split("\n")[:15]
                    console.print("  " + "\n  ".join(lines))
                    if len(snapshot.content.split("\n")) > 15:
                        console.print("  [dim]... (truncated)[/dim]")

            if len(snapshots) > 5:
                console.print(f"\n[dim]... and {len(snapshots) - 5} more files[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def list_commits(
    repo_path: Path = typer.Argument(..., help="Path to Git repository"),
    max_count: int = typer.Option(20, "--max", "-n", help="Maximum commits to show"),
    branch: str = typer.Option("HEAD", "--branch", "-b", help="Branch to list from"),
) -> None:
    """List recent commits in a repository."""
    try:
        config = RepositoryConfig(repo_path=repo_path)
        extractor = GitExtractor(config)

        console.print(f"[bold green]Listing commits from:[/bold green] {repo_path}")
        console.print(f"[bold blue]Branch:[/bold blue] {branch}\n")

        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Hash", style="cyan", width=10)
        table.add_column("Author", style="green")
        table.add_column("Date", style="blue")
        table.add_column("Message", style="white")
        table.add_column("Files", justify="right", style="yellow")

        for commit in extractor.extract_all_commits(branch=branch, max_count=max_count):
            table.add_row(
                commit.short_hash,
                commit.author_name[:20],
                commit.timestamp.strftime("%Y-%m-%d %H:%M"),
                commit.message_summary[:60],
                str(len(commit.files_changed)),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def enrich(
    repo_path: Path = typer.Argument(..., help="Path to Git repository"),
    embedding_provider: str = typer.Option("local", "--embedding-provider", help="Embedding provider: local or openai"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key (required for openai provider)"),
    model: str = typer.Option("gpt-4-turbo-preview", "--model", "-m", help="LLM model for completions (OpenAI only)"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Embedding model name (provider-specific)"),
    max_commits: Optional[int] = typer.Option(None, "--max-commits", "-n", help="Maximum commits to process"),
    branch: str = typer.Option("HEAD", "--branch", "-b", help="Branch to process"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    batch_size: int = typer.Option(10, "--batch-size", help="Number of commits to process concurrently"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Enrich commits with LLM analysis (intent, topics, summary, embeddings).

    Default uses local embeddings (free, privacy-preserving, offline).
    Optionally use OpenAI for maximum quality (~5-7% better).
    """
    try:
        from gitforai.llm import LOCAL_PROVIDER_AVAILABLE
        from gitforai.llm.processor import SemanticProcessor

        # Initialize embedding provider based on selection
        if embedding_provider == "local":
            if not LOCAL_PROVIDER_AVAILABLE:
                console.print(
                    "[bold red]Error:[/bold red] Local embeddings not available. "
                    "Install with: pip install 'gitforai[local-embeddings]'"
                )
                raise typer.Exit(1)

            from gitforai.llm.local_provider import LocalProvider

            # Default to all-MiniLM-L6-v2 for local
            emb_model = embedding_model or "all-MiniLM-L6-v2"
            provider = LocalProvider(model=emb_model)

            console.print(f"[bold green]Enriching commits from:[/bold green] {repo_path}")
            console.print(f"[bold blue]Embedding Provider:[/bold blue] Local (sentence-transformers)")
            console.print(f"[bold blue]Embedding Model:[/bold blue] {emb_model}")
            console.print(f"[bold blue]Cost:[/bold blue] $0.00 (free, runs locally)")

        elif embedding_provider == "openai":
            if not api_key:
                console.print(
                    "[bold red]Error:[/bold red] OpenAI API key required for openai provider. "
                    "Set OPENAI_API_KEY environment variable or use --api-key"
                )
                raise typer.Exit(1)

            from gitforai.llm.openai_provider import OpenAIProvider

            # Default to text-embedding-3-small for OpenAI
            emb_model = embedding_model or "text-embedding-3-small"
            provider = OpenAIProvider(api_key=api_key, model=model, embedding_model=emb_model)

            console.print(f"[bold green]Enriching commits from:[/bold green] {repo_path}")
            console.print(f"[bold blue]Embedding Provider:[/bold blue] OpenAI API")
            console.print(f"[bold blue]LLM Model:[/bold blue] {model}")
            console.print(f"[bold blue]Embedding Model:[/bold blue] {emb_model}")

        else:
            console.print(f"[bold red]Error:[/bold red] Unknown embedding provider: {embedding_provider}")
            console.print("Available providers: local, openai")
            raise typer.Exit(1)

        console.print(f"[bold blue]Caching:[/bold blue] {'Disabled' if no_cache else 'Enabled'}\n")

        # Initialize components
        config = RepositoryConfig(repo_path=repo_path)
        extractor = GitExtractor(config)
        processor = SemanticProcessor(provider, use_cache=not no_cache)

        # Extract commits
        commits = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting commits...", total=None)
            for commit_meta in extractor.extract_all_commits(branch=branch, max_count=max_commits):
                commits.append(commit_meta)
            progress.update(task, completed=True)

        console.print(f"[bold green]✓[/bold green] Extracted {len(commits)} commits\n")

        # Enrich commits
        console.print("[bold yellow]Enriching with LLM analysis...[/bold yellow]")

        async def enrich_async():
            return await processor.enrich_commits_batch(
                commits,
                batch_size=batch_size,
                include_embeddings=True,
            )

        enriched_commits = asyncio.run(enrich_async())

        console.print(f"\n[bold green]✓[/bold green] Enriched {len(enriched_commits)} commits")

        # Show stats
        stats = processor.get_stats()
        if "cache" in stats:
            cache_stats = stats["cache"]
            console.print(f"\n[bold]Cache Stats:[/bold]")
            console.print(f"  Hits: {cache_stats['hits']}")
            console.print(f"  Misses: {cache_stats['misses']}")
            console.print(f"  Hit Rate: {cache_stats['hit_rate']}")

        if "provider" in stats:
            provider_stats = stats["provider"]
            console.print(f"\n[bold]API Usage:[/bold]")
            console.print(f"  Input Tokens: {provider_stats['total_tokens']['input']:,}")
            console.print(f"  Output Tokens: {provider_stats['total_tokens']['output']:,}")
            console.print(f"  Total Cost: ${provider_stats['total_cost']:.4f}")

        # Show examples if verbose
        if verbose and enriched_commits:
            console.print(f"\n[bold]Sample Enriched Commit:[/bold]")
            sample = enriched_commits[0]
            console.print(f"  Hash: [cyan]{sample.short_hash}[/cyan]")
            console.print(f"  Message: {sample.message_summary}")
            console.print(f"  Intent: [yellow]{sample.intent}[/yellow]")
            console.print(f"  Topics: {', '.join(sample.topics)}")
            console.print(f"  Summary: {sample.llm_summary}")
            if sample.embedding:
                console.print(f"  Embedding: {len(sample.embedding)} dimensions")

        # Save to file if output specified
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(
                    [c.model_dump(mode="json") for c in enriched_commits],
                    f,
                    indent=2,
                    default=str,
                )
            console.print(f"\n[bold green]✓[/bold green] Saved to {output}")

    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] LLM dependencies not installed. Install with: pip install gitforai[llm]")
        if verbose:
            console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def semantic_stats(
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Cache directory (defaults to ~/.gitforai/cache)"),
) -> None:
    """Show semantic processing statistics and cache information."""
    try:
        from gitforai.llm.cache import LLMCache

        if cache_dir is None:
            cache_dir = Path.home() / ".gitforai" / "cache"

        if not cache_dir.exists():
            console.print(f"[yellow]Cache directory does not exist:[/yellow] {cache_dir}")
            return

        cache = LLMCache(cache_dir)
        stats = cache.get_stats()

        console.print(f"\n[bold]Cache Statistics[/bold]")
        console.print(f"[cyan]Location:[/cyan] {cache_dir}")
        console.print(f"[cyan]Cached Completions:[/cyan] {stats['cached_completions']}")
        console.print(f"[cyan]Cached Embeddings:[/cyan] {stats['cached_embeddings']}")
        console.print(f"[cyan]Session Hits:[/cyan] {stats['hits']}")
        console.print(f"[cyan]Session Misses:[/cyan] {stats['misses']}")
        console.print(f"[cyan]Hit Rate:[/cyan] {stats['hit_rate']}")

        # Calculate cache size
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*.json"))
        console.print(f"[cyan]Total Size:[/cyan] {total_size / 1024 / 1024:.2f} MB")

    except ImportError:
        console.print("[bold red]Error:[/bold red] LLM dependencies not installed. Install with: pip install gitforai[llm]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def clear_cache(
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Cache directory (defaults to ~/.gitforai/cache)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
) -> None:
    """Clear the LLM cache."""
    try:
        from gitforai.llm.cache import LLMCache

        if cache_dir is None:
            cache_dir = Path.home() / ".gitforai" / "cache"

        if not cache_dir.exists():
            console.print(f"[yellow]Cache directory does not exist:[/yellow] {cache_dir}")
            return

        # Get current stats
        cache = LLMCache(cache_dir)
        stats = cache.get_stats()

        console.print(f"\n[bold]Cache to be cleared:[/bold]")
        console.print(f"  Location: {cache_dir}")
        console.print(f"  Completions: {stats['cached_completions']}")
        console.print(f"  Embeddings: {stats['cached_embeddings']}")

        # Confirm unless forced
        if not force:
            confirm = typer.confirm("\nAre you sure you want to clear the cache?")
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Clear cache
        cache.clear()
        console.print(f"\n[bold green]✓[/bold green] Cache cleared")

    except ImportError:
        console.print("[bold red]Error:[/bold red] LLM dependencies not installed. Install with: pip install gitforai[llm]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to Git repository"),
    embedding_provider: str = typer.Option("local", "--embedding-provider", help="Embedding provider: local or openai"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key"),
    model: str = typer.Option("gpt-4-turbo-preview", "--model", "-m", help="LLM model (OpenAI only)"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Embedding model name"),
    max_commits: Optional[int] = typer.Option(None, "--max-commits", "-n", help="Maximum commits to index"),
    branch: str = typer.Option("HEAD", "--branch", "-b", help="Branch to index"),
    batch_size: int = typer.Option(10, "--batch-size", help="Batch size for processing"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Index repository commits into vector database for semantic search.

    This command:
    1. Extracts commits from the repository
    2. Enriches them with LLM analysis (intent, topics, summary)
    3. Generates embeddings
    4. Stores everything in the vector database for fast semantic search

    Default uses local embeddings (free, privacy-preserving, offline).
    """
    try:
        from gitforai.llm import LOCAL_PROVIDER_AVAILABLE
        from gitforai.llm.processor import SemanticProcessor
        from gitforai.llm.embeddings import EmbeddingService
        from gitforai.storage import VectorStore, CommitDocument

        # Initialize embedding provider
        if embedding_provider == "local":
            if not LOCAL_PROVIDER_AVAILABLE:
                console.print(
                    "[bold red]Error:[/bold red] Local embeddings not available. "
                    "Install with: pip install 'gitforai[local-embeddings]'"
                )
                raise typer.Exit(1)

            from gitforai.llm.local_provider import LocalProvider

            emb_model = embedding_model or "all-MiniLM-L6-v2"
            provider = LocalProvider(model=emb_model)

            console.print(f"[bold green]Indexing commits from:[/bold green] {repo_path}")
            console.print(f"[bold blue]Embedding Provider:[/bold blue] Local (sentence-transformers)")
            console.print(f"[bold blue]Embedding Model:[/bold blue] {emb_model}")
            console.print(f"[bold blue]Cost:[/bold blue] $0.00 (free, runs locally)")

        elif embedding_provider == "openai":
            if not api_key:
                console.print(
                    "[bold red]Error:[/bold red] OpenAI API key required. "
                    "Set OPENAI_API_KEY environment variable or use --api-key"
                )
                raise typer.Exit(1)

            from gitforai.llm.openai_provider import OpenAIProvider

            emb_model = embedding_model or "text-embedding-3-small"
            provider = OpenAIProvider(api_key=api_key, model=model, embedding_model=emb_model)

            console.print(f"[bold green]Indexing commits from:[/bold green] {repo_path}")
            console.print(f"[bold blue]Embedding Provider:[/bold blue] OpenAI API")
            console.print(f"[bold blue]Embedding Model:[/bold blue] {emb_model}")

        else:
            console.print(f"[bold red]Error:[/bold red] Unknown provider: {embedding_provider}")
            raise typer.Exit(1)

        console.print(f"[bold blue]Caching:[/bold blue] {'Disabled' if no_cache else 'Enabled'}\n")

        # Initialize components
        config = RepositoryConfig(repo_path=repo_path)
        extractor = GitExtractor(config)
        processor = SemanticProcessor(provider, use_cache=not no_cache)
        embedding_service = EmbeddingService(provider)
        vector_store = VectorStore()

        # Step 1: Extract commits
        commits = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting commits...", total=None)
            for commit_meta in extractor.extract_all_commits(branch=branch, max_count=max_commits):
                commits.append(commit_meta)
            progress.update(task, completed=True)

        console.print(f"[bold green]✓[/bold green] Extracted {len(commits)} commits\n")

        # Step 2: Enrich commits with LLM analysis
        console.print("[bold yellow]Enriching with LLM analysis...[/bold yellow]")

        async def enrich_and_index():
            # Enrich commits (includes embeddings)
            enriched = await processor.enrich_commits_batch(
                commits,
                batch_size=batch_size,
                include_embeddings=True,
            )

            console.print(f"[bold green]✓[/bold green] Enriched {len(enriched)} commits\n")

            # Step 3: Convert to CommitDocument format and store
            console.print("[bold yellow]Storing in vector database...[/bold yellow]")

            commit_docs = []
            for commit in enriched:
                if not commit.embedding:
                    # Generate embedding if missing
                    commit.embedding = await embedding_service.embed_commit(commit)

                doc = CommitDocument(
                    id=commit.hash,
                    embedding=commit.embedding,
                    author=commit.author_name,
                    author_email=commit.author_email,
                    timestamp=commit.timestamp,
                    message=commit.message,
                    summary=commit.llm_summary,
                    intent=commit.intent,
                    topics=commit.topics,
                    files_changed=commit.files_changed,
                    num_files_changed=len(commit.files_changed),
                    num_lines_added=commit.stats.insertions if commit.stats else 0,
                    num_lines_deleted=commit.stats.deletions if commit.stats else 0,
                    diff_preview=commit.diff_preview,
                    parent_hashes=commit.parent_hashes,
                    is_merge=commit.is_merge,
                    branch=branch,
                    repo_path=str(repo_path),
                )
                commit_docs.append(doc)

            # Upsert to database (insert or update)
            inserted = vector_store.upsert_commits(commit_docs)

            console.print(f"[bold green]✓[/bold green] Indexed {inserted} commits\n")

            # Update state for incremental updates
            if enriched:
                from gitforai.incremental import StateManager, IncrementalUpdateManager
                state_manager = StateManager()
                manager = IncrementalUpdateManager(state_manager, console)

                last_commit = enriched[-1].hash
                manager.update_state(
                    repo_path=repo_path,
                    branch=branch,
                    last_commit=last_commit,
                    commits_processed=len(enriched),
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model or emb_model,
                )

            return enriched

        enriched_commits = asyncio.run(enrich_and_index())

        # Show statistics
        stats = processor.get_stats()
        if "cache" in stats:
            cache_stats = stats["cache"]
            console.print(f"[bold]Cache Stats:[/bold]")
            console.print(f"  Hits: {cache_stats['hits']}")
            console.print(f"  Misses: {cache_stats['misses']}")
            console.print(f"  Hit Rate: {cache_stats['hit_rate']}\n")

        if "provider" in stats:
            provider_stats = stats["provider"]
            # Only show API usage if we have token stats (not available for local provider)
            if "total_tokens" in provider_stats:
                console.print(f"[bold]API Usage:[/bold]")
                console.print(f"  Input Tokens: {provider_stats['total_tokens']['input']:,}")
                console.print(f"  Output Tokens: {provider_stats['total_tokens']['output']:,}")
                console.print(f"  Total Cost: ${provider_stats['total_cost']:.4f}\n")

        # Show database stats
        db_stats = vector_store.get_stats()
        console.print(f"[bold]Vector Database Stats:[/bold]")
        console.print(f"  Total Commits: {db_stats['collections']['commits']['count']}")
        console.print(f"  Provider: {db_stats['config']['provider']}")
        console.print(f"  Persist Dir: {db_stats['config']['persist_dir']}")

        # Show sample if verbose
        if verbose and enriched_commits:
            console.print(f"\n[bold]Sample Indexed Commit:[/bold]")
            sample = enriched_commits[0]
            console.print(f"  Hash: [cyan]{sample.short_hash}[/cyan]")
            console.print(f"  Message: {sample.message_summary}")
            console.print(f"  Intent: [yellow]{sample.intent}[/yellow]")
            console.print(f"  Topics: {', '.join(sample.topics)}")
            console.print(f"  Embedding: {len(sample.embedding)} dimensions")

    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Required dependencies not installed.")
        console.print("Install with: pip install 'gitforai[vectordb,local-embeddings]'")
        if verbose:
            console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(10, "--results", "-n", help="Number of results"),
    intent: Optional[str] = typer.Option(None, "--intent", help="Filter by intent (bug_fix, feature, etc)"),
    author: Optional[str] = typer.Option(None, "--author", help="Filter by author"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Search indexed commits using semantic similarity.

    Examples:
      gitforai search "authentication bug fixes"
      gitforai search "database migration" --intent feature
      gitforai search "security improvements" --author john@example.com
    """
    try:
        from gitforai.llm import LOCAL_PROVIDER_AVAILABLE
        from gitforai.llm.local_provider import LocalProvider
        from gitforai.llm.embeddings import EmbeddingService
        from gitforai.storage import QueryEngine

        # Initialize provider and query engine
        if not LOCAL_PROVIDER_AVAILABLE:
            console.print(
                "[bold red]Error:[/bold red] Local embeddings not available. "
                "Install with: pip install 'gitforai[local-embeddings]'"
            )
            raise typer.Exit(1)

        provider = LocalProvider(model="all-MiniLM-L6-v2")
        embedding_service = EmbeddingService(provider)
        query_engine = QueryEngine(embedding_service=embedding_service)

        console.print(f"[bold green]Searching for:[/bold green] {query}\n")

        # Build filters
        filters = {}
        if intent:
            filters["intent"] = intent
        if author:
            if "@" in author:
                filters["author_email"] = author
            else:
                filters["author"] = author

        # Execute search
        async def search_async():
            return await query_engine.search_commits(query, n_results, filters)

        results = asyncio.run(search_async())

        if not results or len(results) == 0:
            console.print("[yellow]No results found.[/yellow]")
            return

        # Display results
        console.print(f"[bold]Found {len(results)} results:[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Hash", style="cyan", width=10)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Intent", width=12)
        table.add_column("Message", overflow="fold")

        for doc, distance, metadata, commit_id in results:
            # Distance is converted to similarity score (lower distance = higher similarity)
            similarity = max(0, 1 - distance)

            table.add_row(
                commit_id[:8],
                f"{similarity:.3f}",
                metadata.get("intent", "unknown"),
                metadata.get("message", "")[:100],
            )

        console.print(table)

        # Show detailed view if verbose
        if verbose and len(results) > 0:
            console.print(f"\n[bold]Top Result Details:[/bold]")
            doc, distance, metadata, commit_id = list(results)[0]
            console.print(f"  Hash: [cyan]{commit_id}[/cyan]")
            console.print(f"  Author: {metadata.get('author', 'unknown')}")
            console.print(f"  Date: {metadata.get('timestamp', 'unknown')}")
            console.print(f"  Intent: [yellow]{metadata.get('intent', 'unknown')}[/yellow]")
            console.print(f"  Files: {metadata.get('num_files_changed', 0)}")
            console.print(f"  Message: {metadata.get('message', '')}")
            if metadata.get("summary"):
                console.print(f"  Summary: {metadata['summary']}")

    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Required dependencies not installed.")
        console.print("Install with: pip install 'gitforai[vectordb,local-embeddings]'")
        if verbose:
            console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command(name="db-stats")
def db_stats() -> None:
    """Show vector database statistics."""
    try:
        from gitforai.storage import VectorStore

        vector_store = VectorStore()
        stats = vector_store.get_stats()

        console.print("[bold]Vector Database Statistics[/bold]\n")

        # Collections
        console.print("[bold cyan]Collections:[/bold cyan]")
        for name, info in stats["collections"].items():
            console.print(f"  {name}: {info['count']} documents")

        # Configuration
        console.print(f"\n[bold cyan]Configuration:[/bold cyan]")
        console.print(f"  Provider: {stats['config']['provider']}")
        console.print(f"  Persist Dir: {stats['config']['persist_dir']}")
        console.print(f"  Embedding Dim: {stats['config']['embedding_dimension']}")
        console.print(f"  Distance Metric: {stats['config']['distance_metric']}")
        console.print(f"  Batch Size: {stats['config']['batch_size']}")

    except ImportError:
        console.print("[bold red]Error:[/bold red] Vector database dependencies not installed.")
        console.print("Install with: pip install 'gitforai[vectordb]'")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command(name="db-reset")
def db_reset(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Reset vector database (delete all data).

    WARNING: This will permanently delete all indexed commits from the vector database.
    The original git repository is not affected.
    """
    try:
        from gitforai.storage import VectorStore

        if not confirm:
            response = typer.confirm(
                "Are you sure you want to delete all data from the vector database?"
            )
            if not response:
                console.print("[yellow]Cancelled.[/yellow]")
                return

        vector_store = VectorStore()
        vector_store.reset_all()

        console.print("[bold green]✓[/bold green] Vector database reset successfully")

    except ImportError:
        console.print("[bold red]Error:[/bold red] Vector database dependencies not installed.")
        console.print("Install with: pip install 'gitforai[vectordb]'")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def status(
    repo_path: Path = typer.Argument(..., help="Path to Git repository"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch to check (defaults to current)"),
) -> None:
    """Check indexing status for a repository.

    Shows whether the repository has been indexed, last indexed commit,
    number of new commits, and other status information.
    """
    try:
        import git
        from gitforai.incremental import StateManager, IncrementalUpdateManager

        state_manager = StateManager()
        manager = IncrementalUpdateManager(state_manager, console)

        # Get status
        status_info = manager.get_status(repo_path, branch)

        if "error" in status_info:
            console.print(f"[bold red]Error:[/bold red] {status_info['error']}")
            raise typer.Exit(1)

        # Display status
        console.print("\n[bold]Repository Indexing Status[/bold]")
        console.print(f"[cyan]Repository:[/cyan] {repo_path}")
        console.print(f"[cyan]Branch:[/cyan] {status_info.get('current_branch', 'Unknown')}")
        console.print(f"[cyan]Current HEAD:[/cyan] {status_info.get('current_head', 'Unknown')[:8]}")

        if not status_info.get("indexed"):
            console.print("\n[yellow]This repository has not been indexed yet.[/yellow]")
            console.print(f"[yellow]Total commits available:[/yellow] {status_info.get('total_commits', 0)}")
            console.print("\n[dim]Run 'gitforai index <repo>' to index this repository.[/dim]")
        else:
            console.print(f"\n[green]✓ Repository has been indexed[/green]")
            console.print(f"[cyan]Last indexed commit:[/cyan] {status_info.get('last_indexed_commit', 'Unknown')[:8]}")
            console.print(f"[cyan]Last indexed at:[/cyan] {status_info.get('last_indexed_at', 'Unknown')}")
            console.print(f"[cyan]Commits indexed:[/cyan] {status_info.get('commits_indexed', 0)}")
            console.print(f"[cyan]Total commits in branch:[/cyan] {status_info.get('total_commits', 0)}")
            console.print(f"[cyan]Embedding provider:[/cyan] {status_info.get('embedding_provider', 'Unknown')}")

            if status_info.get("embedding_model"):
                console.print(f"[cyan]Embedding model:[/cyan] {status_info['embedding_model']}")

            new_commits = status_info.get("new_commits", 0)
            if status_info.get("rebase_detected"):
                console.print("\n[yellow]⚠ Rebase or force push detected![/yellow]")
                console.print("[yellow]Full reindex is required.[/yellow]")
            elif new_commits > 0:
                console.print(f"\n[yellow]⚠ {new_commits} new commit(s) available[/yellow]")
                console.print("[dim]Run 'gitforai update <repo>' to index new commits.[/dim]")
            else:
                console.print("\n[green]✓ Up to date - no new commits to index[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def update(
    repo_path: Path = typer.Argument(..., help="Path to Git repository"),
    embedding_provider: str = typer.Option("local", "--embedding-provider", help="Embedding provider: local or openai"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key"),
    model: str = typer.Option("gpt-4-turbo-preview", "--model", "-m", help="LLM model (OpenAI only)"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Embedding model name"),
    max_commits: Optional[int] = typer.Option(None, "--max-commits", "-n", help="Maximum commits to index"),
    branch: str = typer.Option("HEAD", "--branch", "-b", help="Branch to update"),
    batch_size: int = typer.Option(10, "--batch-size", help="Batch size for processing"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    force: bool = typer.Option(False, "--force", help="Force full reindex"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Incrementally update the vector database with new commits.

    This command intelligently detects new commits since the last indexing
    and only processes those, avoiding the need to reindex the entire repository.

    If a rebase or force push is detected, it will automatically perform a full reindex.
    Use --force to manually trigger a full reindex.
    """
    try:
        import git
        from gitforai.llm import LOCAL_PROVIDER_AVAILABLE
        from gitforai.llm.processor import SemanticProcessor
        from gitforai.llm.embeddings import EmbeddingService
        from gitforai.storage import VectorStore, CommitDocument
        from gitforai.incremental import StateManager, IncrementalUpdateManager

        # Initialize state manager
        state_manager = StateManager()
        manager = IncrementalUpdateManager(state_manager, console)

        # Initialize Git repo
        try:
            git_repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            console.print(f"[bold red]Error:[/bold red] {repo_path} is not a valid Git repository")
            raise typer.Exit(1)

        # Resolve branch
        if branch == "HEAD":
            branch = git_repo.active_branch.name

        # Initialize embedding provider (same as index command)
        if embedding_provider == "local":
            if not LOCAL_PROVIDER_AVAILABLE:
                console.print(
                    "[bold red]Error:[/bold red] Local embeddings not available. "
                    "Install with: pip install 'gitforai[local-embeddings]'"
                )
                raise typer.Exit(1)

            from gitforai.llm.local_provider import LocalProvider

            emb_model = embedding_model or "all-MiniLM-L6-v2"
            provider = LocalProvider(model=emb_model)

            console.print(f"[bold green]Updating commits from:[/bold green] {repo_path}")
            console.print(f"[bold blue]Embedding Provider:[/bold blue] Local (sentence-transformers)")
            console.print(f"[bold blue]Embedding Model:[/bold blue] {emb_model}")

        elif embedding_provider == "openai":
            if not api_key:
                console.print(
                    "[bold red]Error:[/bold red] OpenAI API key required. "
                    "Set OPENAI_API_KEY environment variable or use --api-key"
                )
                raise typer.Exit(1)

            from gitforai.llm.openai_provider import OpenAIProvider

            emb_model = embedding_model or "text-embedding-3-small"
            provider = OpenAIProvider(api_key=api_key, model=model, embedding_model=emb_model)

            console.print(f"[bold green]Updating commits from:[/bold green] {repo_path}")
            console.print(f"[bold blue]Embedding Provider:[/bold blue] OpenAI API")
            console.print(f"[bold blue]Embedding Model:[/bold blue] {emb_model}")

        else:
            console.print(f"[bold red]Error:[/bold red] Unknown provider: {embedding_provider}")
            raise typer.Exit(1)

        # Check if incremental update should be used
        if not force and manager.should_use_incremental(repo_path, branch, embedding_provider):
            # Get commits to process
            commits_to_process, is_full_reindex = manager.get_commits_to_process(
                git_repo, repo_path, branch, max_commits
            )

            if not commits_to_process:
                console.print("\n[green]✓ Already up to date - no new commits to index[/green]")
                return

            if is_full_reindex:
                console.print("[yellow]Performing full reindex...[/yellow]\n")
            else:
                console.print(f"[green]Performing incremental update ({len(commits_to_process)} new commits)...[/green]\n")
        else:
            if force:
                console.print("[yellow]Force flag set - performing full reindex...[/yellow]\n")
            # Full reindex
            from gitforai.incremental.delta import DeltaDetector
            detector = DeltaDetector(git_repo)
            commits_to_process = detector.get_all_commits(branch, max_count=max_commits)
            is_full_reindex = True

        # Initialize components
        config = RepositoryConfig(repo_path=repo_path)
        extractor = GitExtractor(config)
        processor = SemanticProcessor(provider, use_cache=not no_cache)
        embedding_service = EmbeddingService(provider)
        vector_store = VectorStore()

        console.print(f"[bold yellow]Processing {len(commits_to_process)} commits...[/bold yellow]")

        # Extract, enrich, and store commits
        async def process_and_store():
            # Convert Git commits to CommitMetadata
            commit_metas = []
            for git_commit in commits_to_process:
                commit_meta = extractor._extract_commit_metadata(git_commit)
                commit_metas.append(commit_meta)

            # Enrich with LLM
            enriched = await processor.enrich_commits_batch(
                commit_metas,
                batch_size=batch_size,
                include_embeddings=True,
            )

            console.print(f"[bold green]✓[/bold green] Enriched {len(enriched)} commits\n")

            # Convert to CommitDocument and store
            console.print("[bold yellow]Storing in vector database...[/bold yellow]")

            commit_docs = []
            for commit in enriched:
                if not commit.embedding:
                    commit.embedding = await embedding_service.embed_commit(commit)

                doc = CommitDocument(
                    id=commit.hash,
                    embedding=commit.embedding,
                    author=commit.author_name,
                    author_email=commit.author_email,
                    timestamp=commit.timestamp,
                    message=commit.message,
                    summary=commit.llm_summary,
                    intent=commit.intent,
                    topics=commit.topics,
                    files_changed=commit.files_changed,
                    num_files_changed=len(commit.files_changed),
                    num_lines_added=commit.stats.insertions if commit.stats else 0,
                    num_lines_deleted=commit.stats.deletions if commit.stats else 0,
                    diff_preview=commit.diff_preview,
                    parent_hashes=commit.parent_hashes,
                    is_merge=commit.is_merge,
                    branch=branch,
                    repo_path=str(repo_path),
                )
                commit_docs.append(doc)

            # Upsert to handle both new and existing commits
            vector_store.upsert_commits(commit_docs)
            console.print(f"[bold green]✓[/bold green] Stored {len(commit_docs)} commits\n")

            # Update state
            if commits_to_process:
                last_commit = commits_to_process[-1].hexsha
                manager.update_state(
                    repo_path=repo_path,
                    branch=branch,
                    last_commit=last_commit,
                    commits_processed=len(commit_docs),
                    embedding_provider=embedding_provider,
                    embedding_model=emb_model,
                )

            console.print("[bold green]✓ Update complete![/bold green]")

        asyncio.run(process_and_store())

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from gitforai import __version__

    console.print(f"[bold]GitForAI[/bold] version {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
