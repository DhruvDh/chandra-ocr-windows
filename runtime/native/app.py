"""Serve a NativeBackend through the existing chandra_service routes, admission, streaming and idle policy."""
import asyncio
from contextlib import asynccontextmanager


def create_native_app(backend, **options):
    """chandra_service.create_app plus one addition: shutdown retires the resident worker after the service drains."""
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
