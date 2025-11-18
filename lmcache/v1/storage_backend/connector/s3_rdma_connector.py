# Hewlett Packard Enterprise Confidential
"""RDMA-enabled S3 connector for LMCache."""

# Standard
from typing import Dict, List, Optional
from functools import partial

# Standard library imports
import asyncio
import ctypes
import os
import time
from enum import IntEnum, auto
from itertools import cycle
from threading import Lock

# Third Party
import torch
import mmap
import tempfile
from urllib.parse import quote as url_quote

# Third Party (AWS CRT TCP client for baseline PUT path)
from awscrt import auth, io, s3
from awscrt.http import HttpHeaders, HttpRequest
from awscrt.io import ClientTlsContext, TlsConnectionOptions, TlsContextOptions

# First Party
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# CRITICAL: Must load HPE's cuFile library BEFORE any code imports hpe_object
# This must happen at module import time, not later
_HPE_CUFILE_LOADED = False
_HPE_OBJECT_AVAILABLE = False
_HPE_OBJECT_IMPORT_ERROR = None

def _preload_hpe_cufile():
    """Pre-load HPE's cuFile library to ensure correct version is used."""
    global _HPE_CUFILE_LOADED

    if _HPE_CUFILE_LOADED:
        return True

    hpe_cufile_paths = [
        "/opt/hpe/s3/lib64/libcufile.so.1.13.0",
        "/opt/hpe/s3/lib64/libcufile.so",
    ]

    for path in hpe_cufile_paths:
        if not os.path.exists(path):
            continue

        try:
            # Load with RTLD_GLOBAL to make symbols available globally
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

            # Verify the required symbol exists
            try:
                _ = lib._ZN10cuFileInfo19cuFileGetMemoryTypeEPKv
                _HPE_CUFILE_LOADED = True
                return True
            except AttributeError:
                # Wrong version, try next
                continue

        except Exception:
            continue

    return False

# Pre-load HPE cuFile library FIRST
_cufile_loaded = _preload_hpe_cufile()

# Now try to import hpe_object
try:
    # Third Party
    from hpe_object import (
        BufferGetObject,
        BufferPutObject,
        ClientConfig,
        S3RdmaClient,
    )
    _HPE_OBJECT_AVAILABLE = True
except ImportError as e:
    _HPE_OBJECT_IMPORT_ERROR = str(e)

    if "cuFileGetMemoryType" in str(e):
        _HPE_OBJECT_IMPORT_ERROR = (
            f"\n\nOriginal error: {e}"
        )

# First Party imports
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.s3_rdma_adapter import (
    S3RdmaConnectorSettings
)
# from lmcache.v1.storage_backend.job_executor.pq_executor import (
#     AsyncPQThreadPoolExecutor,
# )
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# Unique prefix for easy log filtering
LOG_PREFIX = "[S3-RDMA]"


class Priorities(IntEnum):
    """
    Priority levels for S3 RDMA operations.
    This enum defines the execution priority order for different types of operations
    in the S3 RDMA connector. Lower numeric values indicate higher priority.
    Attributes:
        GET: Highest priority for cache retrieval operations that block execution
        PUT: Second highest priority for cache storage operations
        PEEK: Medium priority for checking cache entry existence
        PREFETCH: Lowest priority for background prefetching operations
    """
    GET = auto()       # 1 - Cache retrieval (highest priority)
    PEEK = auto()      # 2- Existence checks
    PREFETCH = auto()  # 3 - Background prefetching
    PUT = auto()       # 4 - Cache storage


class S3RdmaConnector(RemoteConnector):
    """
    S3 RDMA connector using HPE's RDMA-enabled S3 client.

    Requirements:
    - HPE cuFile library (v1.13.0) at /opt/hpe/s3/lib64/
    - LD_LIBRARY_PATH must prioritize /opt/hpe/s3/lib64 over CUDA paths

    This ensures proper library load order.
    """

    def __init__(
        self,
        settings: S3RdmaConnectorSettings,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        if not _HPE_OBJECT_AVAILABLE:
            error_msg = (
                f"hpe_object package is required for S3 RDMA connector.\n"
                f"{_HPE_OBJECT_IMPORT_ERROR}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        if not _cufile_loaded:
            logger.warning(
                "Could not pre-load HPE cuFile library. "
                "Ensure you're using the fixed launcher script with correct "
                "LD_LIBRARY_PATH."
            )

        self.settings = settings
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self._boto_client = None
        self._object_size_cache: Dict[str, int] = {}
        self._inflight_sema: Optional[asyncio.Semaphore] = None
        # self._io_executor: Optional[AsyncPQThreadPoolExecutor] = None
        self.pq_executor: Optional[AsyncPQExecutor] = None

        self._prefixed_bucket_path = settings.prefix
        self._effective_parallelism = max(1, settings.max_parallel_requests)

        self._client_pool: List[S3RdmaClient] = []
        self._client_pool_size = max(4, self._effective_parallelism) # Number of clients in the pool
        self._client_iterator: Optional[cycle] = None
        self._client_lock = Lock()

    def post_init(self) -> None:
        """Initialize clients after event loop is set up."""
        super().post_init()

        logger.info(
            "Initializing S3 RDMA client (endpoint=%s, bucket=%s, prefix=%s,"
            "parallelism=%d)",
            self.settings.endpoint,
            self.settings.bucket,
            self._prefixed_bucket_path or "(none)",
            self._effective_parallelism,
        )

        client_config = ClientConfig(
            endpoint=self.settings.endpoint,
            max_parallel_requests=self._effective_parallelism,
        )

        if self.settings.max_segment_size is not None:
            logger.info("Using max segment size: %s bytes", self.settings.max_segment_size)
            client_config.max_segment_size = self.settings.max_segment_size

        #self._client = S3RdmaClient(client_config)
        #logger.info("S3 RDMA client initialized")

        # Create pool of S3RdmaClient instances
        for i in range(self._client_pool_size):
            client = S3RdmaClient(client_config)
            self._client_pool.append(client)
            logger.debug(
                "%s Created S3 RDMA client %d/%d",
                LOG_PREFIX, i + 1, self._client_pool_size)

        # Create round-robin iterator
        self._client_iterator = cycle(self._client_pool)

        logger.info(
            "%s S3 RDMA client pool initialized with %d clients",
            LOG_PREFIX, self._client_pool_size)

        self._inflight_sema = asyncio.Semaphore(self._effective_parallelism)
        # self._io_executor = AsyncPQThreadPoolExecutor(
        #     self.loop, max_workers=self._effective_parallelism
        # )
        self.pq_executor = AsyncPQExecutor(self.loop)
        logger.info("S3 RDMA connector initialization complete")

        # Minimal AWS CRT S3 client init (TCP path) for PUT baseline comparison
        event_loop_group = io.EventLoopGroup(self._effective_parallelism)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        self._credentials_provider = \
            auth.AwsCredentialsProvider.new_default_chain(client_bootstrap)

        tls_opts = None
        try:
            tls_ctx = ClientTlsContext(TlsContextOptions())
            tls_opts = TlsConnectionOptions(tls_ctx)
            tls_opts.set_alpn_list(["h2", "http/1.1"])  # best effort, ignore failures
        except Exception:
            tls_opts = None

        logger.info("Initializing AWS CRT S3 TCP client for baseline PUT path")
        self._tcp_s3_client = s3.S3Client(
            bootstrap=client_bootstrap,
            region="us-east-1",
            credential_provider=self._credentials_provider,
            enable_s3express=False,
            tls_connection_options=tls_opts,
            tls_mode=s3.S3RequestTlsMode.DISABLED,  # non-AWS or custom endpoints
        )

    # Pick the next S3RdmaClient from the pool
    def _get_next_client(self) -> S3RdmaClient:
        """Get next client from pool using round-robin scheduling."""
        assert self._client_iterator is not None

        with self._client_lock:
            return next(self._client_iterator)

    def _make_s3_key(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to S3 object key with optional prefix."""
        key_str = key.to_string()
        if self._prefixed_bucket_path:
            result = f"{self._prefixed_bucket_path}/{key_str}"
        else:
            result = key_str
        return result

    def _format_safe_path(self, s3_key: str) -> str:
        """Create a URL-safe path segment for the CRT client."""
        # Keep the key as-is to preserve '/' in object names
        path = f"/{s3_key}"
        return url_quote(path, safe="/")

    def _tcp_s3_upload(self, s3_key: str, send_path: str):
        """Issue a blocking PUT using AWS CRT similar to S3Connector."""
        headers = HttpHeaders()

        # Construct TCP endpoint: bucket.host:port from http://host:port
        endpoint_stripped = self.settings.endpoint.replace("http://", "").replace("https://", "")
        tcp_endpoint = f"{self.settings.bucket}.{endpoint_stripped}"
        logger.info("%s Using TCP endpoint: %s", LOG_PREFIX, tcp_endpoint)

        headers.add("Host", tcp_endpoint)
        req = HttpRequest("PUT", self._format_safe_path(s3_key), headers)

        done = {"err": None, "status": None}

        def on_done(error=None, status_code=None, **kwargs):
            done["err"] = error
            done["status"] = status_code
            if done["err"] or done["status"] not in (200, 201):
                raise RuntimeError(f"Upload failed in S3RdmaConnector TCP path: {done}")

        s3_req = s3.S3Request(
            client=self._tcp_s3_client,
            type=s3.S3RequestType.PUT_OBJECT,
            request=req,
            operation_name="PutObject",
            send_filepath=send_path,
            credential_provider=self._credentials_provider,
            region=self.settings.region if self.settings.region else "us-east-1",
            on_done=on_done,
        )
        return s3_req

    def _get_object_size_sync(self, s3_key: str) -> Optional[int]:
        """Get object size using S3 HEAD request (synchronous)."""
        if s3_key in self._object_size_cache:
            return self._object_size_cache[s3_key]

        try:
            _client = self._get_next_client()
            size = _client.get_object_size(
                bucket=self.settings.bucket,
                key=s3_key
            )
            self._object_size_cache[s3_key] = size
            return size
        except Exception as e:
            logger.debug("Failed to get size for %s: %s", s3_key, e)
            return None

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in S3."""
        return await self.pq_executor.submit_job(
            self._exists,
            key=key,
            priority=Priorities.PEEK,
        )

    async def _exists(self, key: CacheEngineKey) -> bool:
        """Internal exists implementation."""
        s3_key = self._make_s3_key(key)
        # Use run_in_executor since HPE client is sync
        size = await self.loop.run_in_executor(
            None, self._get_object_size_sync, s3_key
        )
        return size is not None

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronous version of exists."""
        s3_key = self._make_s3_key(key)
        result = self._get_object_size_sync(s3_key) is not None
        return result

    def _get_object_sync(self, s3_key: str, memory_obj: MemoryObj) -> bool:
        """Synchronous RDMA GET operation."""
        try:
            # Get the underlying storage
            assert memory_obj.tensor is not None
            storage = memory_obj.tensor.untyped_storage()

            # Create a ctypes pointer to the storage's data
            # This allows the HPE RDMA client to write directly to GPU memory
            storage_ptr = storage.data_ptr()
            storage_size = storage.nbytes()

            # Create buffer from the storage using ctypes
            buffer = (ctypes.c_ubyte * storage_size).from_address(storage_ptr)

            next_client = self._get_next_client()
            assert next_client is not None

            start_perf = time.perf_counter_ns()
            next_client.get_object_buffers(
                BufferGetObject(
                    bucket=self.settings.bucket,
                    key=s3_key,
                    buffer=memoryview(buffer)
                    )
            )
            end_perf = time.perf_counter_ns()
            perf_duration = end_perf - start_perf
            logger.info(
                "%s RDMA GET completed in %.6f ms: %s. Transfer size: %s",
                LOG_PREFIX, perf_duration / 1_000_000, s3_key, storage_size)

            return True

        except RuntimeError as e:
            # hpe_object raises RuntimeError for various errors
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str or "nosuchkey" in error_str:
                logger.debug("Object not found: %s", s3_key)
                return False
            else:
                logger.error("RDMA GET error for %s: %s", s3_key, e)
                raise

        except Exception as e:
            logger.error("Unexpected error during RDMA GET %s: %s", s3_key, e)
            raise

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get object from S3 using RDMA."""
        return await self.pq_executor.submit_job(
            self._get,
            key=key,
            priority=Priorities.GET,
        )

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Internal get implementation with semaphore handling."""
        s3_key = self._make_s3_key(key)

        try:
            # Size check doesn't need semaphore
            size = await self.loop.run_in_executor(
                None, self._get_object_size_sync, s3_key
            )
            if size is None:
                return None

            # Allocate GPU memory directly for RDMA transfer
            try:
                gpu_device = self.local_cpu_backend.dst_device
                gpu_tensor = torch.empty(
                    size,
                    dtype=torch.uint8,
                    device=gpu_device)

                # Create metadata for the MemoryObj
                metadata = MemoryObjMetadata(
                    shape=self.meta_shape,
                    dtype=self.meta_dtype,
                    address=gpu_tensor.data_ptr(),
                    phy_size=size,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.KV_2LTD
                )

                # Create TensorMemoryObj wrapping the GPU tensor
                memory_obj = TensorMemoryObj(
                    raw_data=gpu_tensor,
                    metadata=metadata,
                    parent_allocator=None
                )

            except Exception as e:
                logger.error(
                    "Failed to allocate GPU memory for %s: %s", s3_key, e)
                return None

            # Verify memory location before RDMA transfer
            if not (hasattr(memory_obj, 'tensor') and \
                    memory_obj.tensor is not None):
                logger.error(
                    "%s memory_obj has no tensor attribute!", LOG_PREFIX)
                return None

            if not memory_obj.tensor.is_cuda:
                logger.error(
                    "%s Allocated memory is NOT on GPU! Device: %s. Cannot "
                    "proceed with RDMA transfer.",
                    LOG_PREFIX, memory_obj.tensor.device
                )
                return None

            # Semaphore handling for RDMA operation
            assert self._inflight_sema is not None
            await self._inflight_sema.acquire()
            try:
                success = await self.loop.run_in_executor(
                    None,
                    self._get_object_sync,
                    s3_key,
                    memory_obj)

                if not success:
                    # Clean up on failure
                    memory_obj.invalidate()
                    del memory_obj
                    del gpu_tensor
                    return None

                return memory_obj
            finally:
                self._inflight_sema.release()

        except Exception as e:
            logger.error("Failed to get %s (outer error): %s", s3_key, e, exc_info=True)
            raise

    def _put_object_sync(self, s3_key: str, memory_obj: MemoryObj) -> None:
        """Synchronous RDMA PUT operation."""
        try:
            buffer_view = memory_obj.byte_array
            _client = self._get_next_client()
            assert _client is not None

            _start = time.perf_counter_ns()
            _client.put_object_buffers(
                BufferPutObject(
                    bucket=self.settings.bucket,
                    key=s3_key,
                    buffer=buffer_view
                )
            )
            _end = time.perf_counter_ns()
            _duration_ms = _end - _start
            logger.info(
                "%s RDMA PUT completed in %.6f ms: %s. Transfer size: %s",
                LOG_PREFIX, _duration_ms / 1_000_000,
                s3_key,
                len(buffer_view))

            # Cache the size
            self._object_size_cache[s3_key] = len(buffer_view)

        except RuntimeError as e:
            logger.error("RDMA PUT error for %s: %s", s3_key, e)
            raise
        except Exception as e:
            logger.error("Unexpected error during RDMA PUT %s: %s", s3_key, e)
            raise

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Blocking TCP-like PUT (baseline) with staging copy, submitted via PQ.

        Original (RDMA fire-and-forget) implementation retained below for reference:
        # async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        #     return await self.pq_executor.submit_job(
        #         self._put,
        #         key=key,
        #         memory_obj=memory_obj,
        #         priority=Priorities.PUT,
        #     )
        """
        return await self.pq_executor.submit_job(
            self._put,
            key=key,
            memory_obj=memory_obj,
            priority=Priorities.PUT,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Internal blocking PUT implementation with ephemeral host staging.

        Original (RDMA fire-and-forget) implementation retained below for reference:
        # async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        #     s3_key = self._make_s3_key(key)
        #     self.loop.run_in_executor(
        #         None,
        #         self._put_object_sync,
        #         s3_key,
        #         memory_obj
        #     )
        """
        s3_key = self._make_s3_key(key)

        # Assertions similar to TCP connector
        assert memory_obj.get_physical_size() == self.full_chunk_size, (
            "Saving unfull chunk is not supported in S3RdmaConnector TCP-like PUT"
        )

        await self._inflight_sema.acquire()

        send_tmp = None
        mm = None
        try:
            size_bytes = memory_obj.get_physical_size()

            # Start of perf measurement
            start_perf = time.perf_counter_ns()
            
            # Ephemeral /dev/shm staging file
            send_tmp = tempfile.NamedTemporaryFile(
                prefix="rdma_put_", suffix=".part", dir="/dev/shm", delete=False
            )
            os.ftruncate(send_tmp.fileno(), size_bytes)

            with open(send_tmp.name, "r+b") as f:
                mm = mmap.mmap(f.fileno(), size_bytes)
                buf = ctypes.c_char.from_buffer(mm)
                host_addr = ctypes.addressof(buf)

            # Staging copy timing
            src_ptr = memory_obj.data_ptr
            ctypes.memmove(host_addr, src_ptr, size_bytes)

            s3_req = self._tcp_s3_upload(s3_key, send_tmp.name)
            await asyncio.wrap_future(s3_req.finished_future)
            
            # End of perf measurement
            end_perf = time.perf_counter_ns()

            perf_duration = end_perf - start_perf

            # Update object size cache (mirroring TCP behavior)
            self._object_size_cache[s3_key] = size_bytes

            # Summary log line (mirrors TCP wording with RDMA prefix)
            logger.info("%s Starting TCP-like Upload", LOG_PREFIX)
            logger.info(
                "%s RDMA PUT completed in %.6f ms: %s. Transfer size: %s",
                LOG_PREFIX,
                perf_duration / 1_000_000,
                s3_key,
                size_bytes,
            )

        except Exception as e:
            logger.error("Failed TCP-like PUT for %s: %s", s3_key, e)
            raise
        finally:
            self._inflight_sema.release()
            if mm is not None:
                mm.close()
            if send_tmp is not None:
                try:
                    os.unlink(send_tmp.name)
                except FileNotFoundError:
                    pass

    async def list(self) -> List[str]:
        """List all objects."""
        raise NotImplementedError

    async def close(self) -> None:
        """Clean up resources."""
        if self.pq_executor:
            try:
                self.pq_executor.shutdown(wait=True)
            except Exception as e:
                logger.warning("Error shutting down executor: %s", e)

        # Clean up client pool
        self._client_pool.clear()
        self._client_iterator = None

        self._object_size_cache.clear()
