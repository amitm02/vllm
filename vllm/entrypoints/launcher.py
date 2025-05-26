# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
import socket
from http import HTTPStatus
from typing import Any, Optional

from hypercorn.asyncio import serve as hypercorn_serve
from hypercorn.config import Config as HypercornConfig
from fastapi import FastAPI, Request, Response

from vllm import envs
from vllm.engine.async_llm_engine import AsyncEngineDeadError
from vllm.engine.multiprocessing import MQEngineDeadError
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.ssl import SSLCertRefresher
from vllm.logger import init_logger
from vllm.utils import find_process_using_port
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)


async def serve_http(app: FastAPI,
                     sock: Optional[socket.socket],
                     enable_ssl_refresh: bool = False,
                     **server_kwargs: Any):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ', '.join(methods))

    hypercorn_config = HypercornConfig()
    host = server_kwargs.get('host', '0.0.0.0')
    port = server_kwargs.get('port', 8000)
    if sock:
        hypercorn_config.bind = [sock.getsockname()]
        # Hypercorn's serve function can also take a list of sockets directly
        # We will pass the socket to hypercorn_serve if provided.
    else:
        hypercorn_config.bind = [f"{host}:{port}"]

    hypercorn_config.keyfile = server_kwargs.get('ssl_keyfile')
    hypercorn_config.certfile = server_kwargs.get('ssl_certfile')
    hypercorn_config.ca_certs = server_kwargs.get('ssl_ca_certs')
    # Ensure log level names are compatible, e.g. uvicorn's "debug" to hypercorn's "DEBUG"
    hypercorn_config.loglevel = server_kwargs.get('log_level', 'info').upper()
    # Configure access and error logs. Using "-" for stdout/stderr by default.
    hypercorn_config.accesslog = server_kwargs.get('access_log', "-")
    hypercorn_config.errorlog = "-" # Hypercorn uses errorlog for its own errors
    hypercorn_config.keep_alive_timeout = server_kwargs.get('timeout_keep_alive', 5)
    hypercorn_config.alpn_protocols = ["h2", "http/1.1"]

    # Create SSL context if SSL files are provided
    if hypercorn_config.keyfile and hypercorn_config.certfile:
        hypercorn_config.ssl_context = SSLCertRefresher.create_ssl_context(
            keyfile=hypercorn_config.keyfile,
            certfile=hypercorn_config.certfile,
            ca_certs=hypercorn_config.ca_certs,
        )
    else:
        hypercorn_config.ssl_context = None


    loop = asyncio.get_running_loop()
    # shutdown_trigger is a Future that hypercorn_serve will wait on.
    # Setting a result or cancelling this future will trigger shutdown.
    shutdown_event = asyncio.Event()


    # The server_task will run hypercorn_serve
    # Pass sock directly to hypercorn_serve if it exists
    server_task = loop.create_task(
        hypercorn_serve(app,
                        hypercorn_config,
                        sockets=[sock] if sock else None,
                        shutdown_trigger=shutdown_event.wait # type: ignore
                       )
    )

    # Store shutdown_event in app.state so it can be accessed by exception handlers
    app.state.shutdown_event = shutdown_event
    _add_shutdown_handlers(app, server_task) # Pass server_task for cancellation

    watchdog_task = loop.create_task(
        watchdog_loop(server_task, app.state.engine_client, shutdown_event))


    ssl_cert_refresher = None
    if enable_ssl_refresh and hypercorn_config.ssl_context:
        ssl_cert_refresher = SSLCertRefresher(
            ssl_context=hypercorn_config.ssl_context, # Use Hypercorn's SSL context
            key_path=hypercorn_config.keyfile,
            cert_path=hypercorn_config.certfile,
            ca_path=hypercorn_config.ca_certs)
        # Hypercorn doesn't have a direct way to update SSL context on the fly like Uvicorn's `server.config.ssl = new_context`.
        # For Hypercorn, a restart is typically required for SSL cert changes.
        # The SSLCertRefresher here would create a new context, but applying it
        # to a running Hypercorn server is non-trivial without deeper integration
        # or a restart mechanism.
        # For now, we initialize it, but note that live refresh might not work without further changes.
        logger.info("SSLCertRefresher initialized. Live refresh with Hypercorn may require server restart.")


    def signal_handler() -> None:
        logger.info("Received signal, initiating shutdown.")
        shutdown_event.set() # Trigger Hypercorn's graceful shutdown
        # Cancelling tasks can help ensure they stop if shutdown_event isn't enough
        if not server_task.done():
            server_task.cancel()
        if not watchdog_task.done():
            watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
    except asyncio.CancelledError:
        # This can happen if signal_handler cancels server_task directly,
        # or if another part of the system cancels it.
        logger.info("Hypercorn server task cancelled. Shutting down.")
    except Exception as e:
        logger.error(f"Hypercorn server failed: {e}", exc_info=True)
    finally:
        logger.info("Hypercorn server shutdown process finalizing.")
        if not watchdog_task.done():
            watchdog_task.cancel()
        # Ensure tasks are awaited to prevent warnings, or handled if they error
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass # Expected
        except Exception as e:
            logger.error(f"Error in watchdog task during shutdown: {e}")

        if ssl_cert_refresher:
            ssl_cert_refresher.stop()
        # Additional cleanup if necessary
        logger.info("FastAPI HTTP server shutdown complete.")


async def watchdog_loop(server_task: asyncio.Task, engine: EngineClient, shutdown_event: asyncio.Event):
    """
    Watchdog task that runs in the background, checking
    for error state in the engine. If an error is found,
    it triggers the shutdown of the Hypercorn server.
    """
    VLLM_WATCHDOG_TIME_S = 5.0
    while not shutdown_event.is_set():
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        if shutdown_event.is_set(): # Check again after sleep
            break
        terminate_if_errored(server_task, engine, shutdown_event)


def terminate_if_errored(server_task: asyncio.Task, engine: EngineClient, shutdown_event: asyncio.Event):
    """
    If the engine is in an error state, trigger the shutdown of the
    Hypercorn server by setting the shutdown_event.
    """
    engine_errored = engine.errored and not engine.is_running
    if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:
        if not shutdown_event.is_set():
            logger.warning("Engine errored. Initiating server shutdown via watchdog.")
            shutdown_event.set()
        # Additionally, cancel the server_task as a more direct way to stop it.
        # This is important if hypercorn_serve doesn't immediately react to shutdown_trigger
        # or if the event loop is blocked.
        if not server_task.done():
            server_task.cancel()


def _add_shutdown_handlers(app: FastAPI, server_task: asyncio.Task) -> None:
    """
    VLLM V1 AsyncLLM catches exceptions and returns
    only two types: EngineGenerateError and EngineDeadError.
    
    EngineGenerateError is raised by the per request generate()
    method. This error could be request specific (and therefore
    recoverable - e.g. if there is an error in input processing).
    
    EngineDeadError is raised by the background output_handler
    method. This error is global and therefore not recoverable.
    
    We register these @app.exception_handlers to return nice
    responses to the end user if they occur and shut down if needed.
    See https://fastapi.tiangolo.com/tutorial/handling-errors/
    for more details on how exception handlers work.

    If an exception is encountered in a StreamingResponse
    generator, the exception is not raised, since we already sent
    a 200 status. Rather, we send an error message as the next chunk.
    Since the exception is not raised, this means that the server
    will not automatically shut down. Instead, we use the watchdog
    background task for check for errored state.
    """

    # Note: The original _add_shutdown_handlers passed `server` (Uvicorn server instance)
    # to `terminate_if_errored`. Now we pass `server_task` (Hypercorn's task)
    # and `shutdown_event` (if using event-based shutdown for Hypercorn).
    # For simplicity, we'll stick to cancelling the server_task as the primary
    # mechanism for termination from these handlers, assuming `terminate_if_errored`
    # is adapted to use it.

    # We need to ensure `terminate_if_errored` can be called correctly.
    # It now needs server_task and shutdown_event.
    # The request object (FastAPI) has `app.state.engine_client`.
    # We need to get a reference to shutdown_event if it's used by terminate_if_errored.
    # Let's assume terminate_if_errored is adapted to primarily use server_task.cancel().
    # The `shutdown_event` is created in `serve_http`. If `terminate_if_errored` needs it,
    # it would have to be passed down or accessed globally/via contextvar, which is complex.
    # Simpler: `terminate_if_errored` primarily cancels `server_task`.

    # A simple way to get shutdown_event if needed by terminate_if_errored
    # is to attach it to the app state, though this is a bit of a hack.
    # For now, let's assume terminate_if_errored is modified to primarily use server_task.cancel().
    # The `shutdown_event` is mainly for Hypercorn's own graceful signal handling.

    @app.exception_handler(RuntimeError)
    @app.exception_handler(AsyncEngineDeadError)
    @app.exception_handler(MQEngineDeadError)
    @app.exception_handler(EngineDeadError)
    @app.exception_handler(EngineGenerateError)
    async def runtime_exception_handler(request: Request, exc: Exception):
        # Get the shutdown_event from the app state if it was stored there
        # Or, ensure terminate_if_errored can work with just server_task
        # For this example, let's assume terminate_if_errored is adapted
        # such that it can get the shutdown_event if needed, or primarily relies on task cancellation.
        # The current `terminate_if_errored` signature in this diff expects `shutdown_event`.
        # We need to retrieve it. A common way is to attach it to `app.state`
        # if it's not easily passed down through layers of function calls.
        # However, the `_add_shutdown_handlers` is called from `serve_http` where `shutdown_event` is available.
        # Let's refine this.

        # Simplification: The exception handler will call `terminate_if_errored`,
        # which will cancel `server_task`. The `shutdown_event` is mostly for
        # Hypercorn's internal graceful shutdown on SIGINT/SIGTERM.
        # The watchdog or an exception can more forcefully cancel the server_task.

        engine_client = request.app.state.engine_client
        # We need to get the shutdown_event that was created in serve_http
        # One way is to pass it to _add_shutdown_handlers.
        # Let's adjust the signature of _add_shutdown_handlers and how it's called.
        # For now, assuming terminate_if_errored is called with the right args.
        # This part needs careful wiring of shutdown_event or relying on task cancellation.

        # If terminate_if_errored is defined as:
        # def terminate_if_errored(server_task: asyncio.Task, engine: EngineClient, shutdown_event: asyncio.Event):
        # We need to pass shutdown_event here.
        # Let's assume for now that `_add_shutdown_handlers` is called with `shutdown_event`
        # and can pass it to `terminate_if_errored`.
        # This means modifying the call to `_add_shutdown_handlers(app, server_task, shutdown_event)`

        # Corrected call within the exception handler:
        # This requires shutdown_event to be available here.
        # If _add_shutdown_handlers has access to shutdown_event:
        shutdown_event = app.state.shutdown_event # Assuming it's stored in app.state

        terminate_if_errored(
            server_task=server_task, # server_task is available in this scope from the outer function
            engine=engine_client,
            shutdown_event=shutdown_event
        )

        # Log the actual exception that was caught
        logger.error(f"Unhandled exception in FastAPI: {exc}", exc_info=exc)
        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

# Modify the call to _add_shutdown_handlers in serve_http to pass shutdown_event
# e.g., app.state.shutdown_event = shutdown_event
# _add_shutdown_handlers(app, server_task)
# And then in _add_shutdown_handlers, it can be retrieved from app.state.
# This seems like a common pattern for FastAPI.
# Let's ensure app.state.shutdown_event is set in serve_http before calling _add_shutdown_handlers.
