# Hewlett Packard Enterprise Confidential
"""RDMA-enabled S3 connector for LMCache."""

# Standard
from typing import Dict, List, Optional

# Standard library imports
import asyncio
import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Third Party
import torch

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
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# Unique prefix for easy log filtering
LOG_PREFIX = "[S3-RDMA]"


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

        self._client: Optional[S3RdmaClient] = None
        # self._client_lock = threading.Lock()
        self._boto_client = None
        self._object_size_cache: Dict[str, int] = {}
        self._inflight_sema: Optional[asyncio.Semaphore] = None
        # self._io_executor: Optional[AsyncPQThreadPoolExecutor] = None
        self._io_executor: Optional[ThreadPoolExecutor] = None

        self._prefixed_bucket_path = settings.prefix
        self._effective_parallelism = max(1, settings.max_parallel_requests)

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

        self._client = S3RdmaClient(client_config)
        logger.info("S3 RDMA client initialized")

        self._inflight_sema = asyncio.Semaphore(self._effective_parallelism)
        # self._io_executor = AsyncPQThreadPoolExecutor(
        #     self.loop, max_workers=self._effective_parallelism
        # )
        self._io_executor = ThreadPoolExecutor(
            max_workers=self._effective_parallelism)
        logger.info("S3 RDMA connector initialization complete")

    def _make_s3_key(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to S3 object key with optional prefix."""
        key_str = key.to_string()
        if self._prefixed_bucket_path:
            result = f"{self._prefixed_bucket_path}/{key_str}"
        else:
            result = key_str
        return result

    def _get_object_size_sync(self, s3_key: str) -> Optional[int]:
        """Get object size using S3 HEAD request (synchronous)."""
        if s3_key in self._object_size_cache:
            return self._object_size_cache[s3_key]

        try:
            size = self._client.get_object_size(
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
        s3_key = self._make_s3_key(key)
        size = await self.loop.run_in_executor(
            self._io_executor,
            self._get_object_size_sync,
            s3_key)
        result = size is not None
        return result

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronous version of exists."""
        s3_key = self._make_s3_key(key)
        result = self._get_object_size_sync(s3_key) is not None
        return result

    def _get_object_sync(self, s3_key: str, memory_obj: MemoryObj) -> bool:
        """Synchronous RDMA GET operation."""
        try:
            # Get the underlying storage
            storage = memory_obj.tensor.untyped_storage()

            # Create a ctypes pointer to the storage's data
            # This allows the HPE RDMA client to write directly to GPU memory
            storage_ptr = storage.data_ptr()
            storage_size = storage.nbytes()

            # Create buffer from the storage using ctypes
            buffer = (ctypes.c_ubyte * storage_size).from_address(storage_ptr)

            assert self._client is not None

            _start = time.perf_counter_ns()
            self._client.get_object_buffers(
                BufferGetObject(
                    bucket=self.settings.bucket,
                    key=s3_key,
                    buffer=memoryview(buffer)
                    )
            )
            _end = time.perf_counter_ns()
            _duration_ms = (_end - _start)
            logger.info("%s RDMA GET completed in %.6f ms: %s. Transfer size: %s", LOG_PREFIX, _duration_ms / 1_000_000, s3_key, storage_size)

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
        s3_key = self._make_s3_key(key)

        try:
            size = await self.loop.run_in_executor(
                self._io_executor,
                self._get_object_size_sync,
                s3_key)
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
                logger.error("Failed to allocate GPU memory for %s: %s", s3_key, e)
                return None

            # Verify memory location before RDMA transfer
            if not (hasattr(memory_obj, 'tensor') and memory_obj.tensor is not None):
                logger.error("%s memory_obj has no tensor attribute!", LOG_PREFIX)
                return None

            if not memory_obj.tensor.is_cuda:
                logger.error(
                    "%s Allocated memory is NOT on GPU! Device: %s. Cannot proceed with RDMA transfer.",
                    LOG_PREFIX, memory_obj.tensor.device
                )
                return None

            try:
                assert self._inflight_sema is not None

                async with self._inflight_sema:
                    success = await self.loop.run_in_executor(
                        self._io_executor,
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

            except Exception as e:
                # Clean up on error
                memory_obj.invalidate()
                del memory_obj
                del gpu_tensor
                logger.error("Failed to get %s: %s", s3_key, e, exc_info=True)
                raise

        except Exception as e:
            logger.error("Failed to get %s (outer error): %s", s3_key, e, exc_info=True)
            raise

    def _put_object_sync(self, s3_key: str, memory_obj: MemoryObj) -> None:
        """Synchronous RDMA PUT operation."""
        try:
            buffer_view = memory_obj.byte_array
            assert self._client is not None

            _start = time.perf_counter_ns()
            self._client.put_object_buffers(
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
        """Put object to S3 using RDMA."""
        s3_key = self._make_s3_key(key)

        assert self._inflight_sema is not None

        async with self._inflight_sema:
            await self.loop.run_in_executor(
                self._io_executor,
                self._put_object_sync,
                s3_key,
                memory_obj)

    async def list(self) -> List[str]:
        """List all objects."""
        raise NotImplementedError

    async def close(self) -> None:
        """Clean up resources."""
        if self._io_executor:
            try:
                self._io_executor.shutdown(wait=True)
            except Exception as e:
                logger.warning("Error shutting down executor: %s", e)
        self._object_size_cache.clear()
