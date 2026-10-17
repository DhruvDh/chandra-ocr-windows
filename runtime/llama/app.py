"""The existing service protocol with explicit owned-engine shutdown."""
import asyncio
from contextlib import asynccontextmanager


def create_llama_app(backend, **options):
    from chandra_service.service import create_app
    app = create_app(backend, **options)
    inner = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with inner(application):
                yield
        finally:
            await asyncio.to_thread(backend.close)

    app.router.lifespan_context = lifespan
    return app
