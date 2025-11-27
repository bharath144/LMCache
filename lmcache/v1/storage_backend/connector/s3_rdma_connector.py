# Hewlett Packard Enterprise Confidential
"""RDMA-enabled S3 connector for LMCache."""

# Standard
from typing import Dict, List, Optional

# Standard library imports
import asyncio
import ctypes
import mmap
import os
import tempfile
#import time
from enum import IntEnum, auto

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


class AdhocSharedMemoryManager:
    """
    A shared memory manager that allocates shared memory buffers
    on demand.
    """

    def __init__(
        self,
        shm_buffers: list[int],
        shm_names: list[str],
        mmaps: list[mmap.mmap],
    ):
        self.shm_buffers = shm_buffers
        self.shm_names = shm_names
        self.mmaps = mmaps

    def allocate(self) -> tuple[str, int, mmap.mmap]:
        """
        Allocate a shared memory buffer and return its name and a bytearray
        that can be used to access the buffer.
        """
        if not self.shm_buffers:
            raise RuntimeError("No more shared memory buffers available")

        shm = self.shm_buffers.pop()
        shm_name = self.shm_names.pop()
        mm = self.mmaps.pop()
        return shm_name, shm, mm

    def free(
        self,
        shm_name: str,
        shm: int,
        mm: mmap.mmap,
    ) -> None:
        """
        Free a shared memory buffer.
        """

        self.shm_buffers.append(shm)
        self.shm_names.append(shm_name)
        self.mmaps.append(mm)

    def close(self):
        # let python GC clean up mmap inodes
        for mm in self.mmaps:
            mm.close()
        for shm_name in self.shm_names:
            try:
                os.unlink(shm_name)
            except FileNotFoundError:
                pass  # file probably already removed


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
        self.object_size_cache: Dict[str, int] = {}
        self.pq_executor: Optional[AsyncPQExecutor] = None

        self._prefixed_bucket_path = settings.prefix
        self._effective_parallelism = max(1, settings.max_parallel_requests)

        self.client_pool: List[S3RdmaClient] = []
        self.client_pool_size = 16 # Number of clients in the pool

        self.adhoc_shm_manager: Optional[AdhocSharedMemoryManager] = None

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

        prefix = "s3rdma_shm"
        shms = []
        shm_names = []
        mmaps = []
        for i in range(64):
            shm_name = f"{prefix}_{i}"

            shm = tempfile.NamedTemporaryFile(
                prefix=shm_name, suffix=".part", dir="/dev/shm", delete=False
            )

            os.ftruncate(shm.fileno(), self.full_chunk_size)

            with open(shm.name, "r+b") as f:
                mm = mmap.mmap(f.fileno(), self.full_chunk_size)
                # create a char buffer view over the mmap
                buf = ctypes.c_char.from_buffer(mm)
                addr = ctypes.addressof(buf)

            shms.append(addr)
            shm_names.append(shm.name)
            mmaps.append(mm)

        self.adhoc_shm_manager = AdhocSharedMemoryManager(
            shm_buffers=shms,
            shm_names=shm_names,
            mmaps=mmaps,
        )

        client_config = ClientConfig(
            endpoint=self.settings.endpoint,
            max_parallel_requests=self._effective_parallelism,
        )

        if self.settings.max_segment_size is not None:
            logger.info(
                "Using max segment size: %s bytes",
                self.settings.max_segment_size)
            client_config.max_segment_size = self.settings.max_segment_size

        # Create pool of S3RdmaClient instances
        for i in range(self.client_pool_size):
            client = S3RdmaClient(client_config)
            self.client_pool.append(client)
            logger.debug(
                "%s Created S3 RDMA client %d/%d",
                LOG_PREFIX, i + 1, self.client_pool_size)

        logger.info(
            "%s S3 RDMA client pool initialized with %d clients",
            LOG_PREFIX, self.client_pool_size)

        self.pq_executor = AsyncPQExecutor(self.loop)
        logger.info("S3 RDMA connector initialization complete")

    def _make_s3_key(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to S3 object key with optional prefix."""
        key_str = key.to_string()
        if self._prefixed_bucket_path:
            result = f"{self._prefixed_bucket_path}/{key_str}"
        else:
            result = key_str
        return result

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
        return size != 0

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronous version of exists."""
        s3_key = self._make_s3_key(key)

        size = self._get_object_size_sync(s3_key)

        return size != 0

    def _get_object_size_sync(self, s3_key: str) -> int:
        """Get object size using S3 HEAD request (synchronous)."""
        if s3_key in self.object_size_cache:
            size = self.object_size_cache[s3_key]
        else:
            try:
                client = self.client_pool[hash(s3_key) % self.client_pool_size]
                size = client.get_object_size(
                    bucket=self.settings.bucket,
                    key=s3_key
                )
                self.object_size_cache[s3_key] = size
            except Exception as e:
                logger.debug("Failed to get size for %s: %s", s3_key, e)
                size = 0

        return size

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get object from S3 using RDMA."""
        return await self.pq_executor.submit_job(
            self._get,
            key=key,
            priority=Priorities.GET,
        )

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Internal get implementation with semaphore handling."""
        key_str = self._make_s3_key(key)

        obj_size = self.object_size_cache.get(key_str, None)

        if obj_size is None:
            obj_size = await self.loop.run_in_executor(
                None, self._get_object_size_sync, key_str)

            if obj_size <= 0:
                self.object_size_cache[key_str] = 0
                return None

            self.object_size_cache[key_str] = obj_size

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shape,
            self.meta_dtype,
            self.meta_fmt,
        )

        # storage = memory_obj.tensor.untyped_storage()
        # storage_ptr = storage.data_ptr()
        # buffer = (ctypes.c_ubyte * obj_size).from_address(storage_ptr)

        # object_buffer = BufferGetObject(
        #     bucket=self.settings.bucket, key=key_str,
        #     buffer=memoryview(buffer))

        recv, shm, mm = self.adhoc_shm_manager.allocate()
        src_mview = memoryview(mm)

        object_buffer = BufferGetObject(
            bucket=self.settings.bucket, key=key_str, buffer=src_mview)

        # start_perf = time.perf_counter_ns()

        result = await self.loop.run_in_executor(
            None, self._get_objects_sync, [key_str], [object_buffer])

        # end_perf = time.perf_counter_ns()
        # perf_duration = end_perf - start_perf
        # logger.info(
        #     "%s RDMA GET (individual) completed in %.6f ms: %s",
        #     LOG_PREFIX, perf_duration / 1_000_000, key_str)

        if result:
            # Copy data from shared memory buffer to memory object
            dst_ptr = memory_obj.data_ptr

            # Cast the memoryview into a ctypes array type
            source_bfr = (ctypes.c_ubyte * obj_size).from_buffer(src_mview)
            source_addr = ctypes.addressof(source_bfr)
            ctypes.memmove(dst_ptr, source_addr, obj_size)
        else:
            memory_obj.invalidate()
            del memory_obj
            memory_obj = None

        # Final cleanup, irrespective of success or failure
        self.adhoc_shm_manager.free(recv, shm, mm)

        return memory_obj

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Batched get implementation for RDMA"""
        return await self.pq_executor.submit_job(
            self._batched_get,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    async def _batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Internal implementation of batched get for RDMA"""

        memory_objs: List[Optional[MemoryObj]] = []
        keys_list: List[str] = []
        buffer_objects: List[BufferGetObject] = []
        resources: List[tuple[str, int, mmap.mmap, memoryview]] = []

        # Prepare all MemoryObjects and allocate resources
        for key in keys:
            key_str = self._make_s3_key(key)

            # First check if we have cached object size
            obj_size = self.object_size_cache.get(key_str, None)

            # If we don't have the size cached, make a HEAD_OBJECT call
            if obj_size is None:
                obj_size = await self.loop.run_in_executor(
                    None, self._get_object_size_sync, key_str)

                if obj_size <= 0:
                    self.object_size_cache[key_str] = 0
                    continue

                self.object_size_cache[key_str] = obj_size

            keys_list.append(key_str)

            mem_obj = self.local_cpu_backend.allocate(
                self.meta_shape,
                self.meta_dtype,
                self.meta_fmt,
            )

            # Append to the list of memory objects
            memory_objs.append(mem_obj)

            if not mem_obj:
                continue

            # Allocate shared memory buffer
            recv, shm, mm = self.adhoc_shm_manager.allocate()
            src_view = memoryview(mm)

            # Store resources for cleanup later
            resources.append((recv, shm, mm, src_view))

            buffer_object = BufferGetObject(
                bucket=self.settings.bucket, key=key_str, buffer=src_view)

            buffer_objects.append(buffer_object)

        if buffer_objects:
            # start_perf = time.perf_counter_ns()

            # Now we are ready to call synchronous get on all the objects
            status = await self.loop.run_in_executor(
                None, self._get_objects_sync, keys_list, buffer_objects)

            # end_perf = time.perf_counter_ns()
            # perf_duration = end_perf - start_perf
            # logger.info(
            #     "%s RDMA GET (batched) completed in %.6f ms for %d objects",
            #     LOG_PREFIX, perf_duration / 1_000_000,
            #     len(buffer_objects))

            if not status:
                # Unlikely situation, just logging a message here for now.
                # Better error handling needs to be implemented.
                # TODO: Implement better error handling for batched_get
                logger.warning(
                    "%s Batched GET encountered errors. Some objects may be "
                    "missing.", LOG_PREFIX)

            # Copy data from shared memory buffers to memory objects
            resource_idx = 0
            for i, memory_obj in enumerate(memory_objs):
                if memory_obj is None:
                    continue

                recv, shm, mm, src_view = resources[resource_idx]
                key_str = keys[i].to_string()
                obj_size = self.object_size_cache[key_str]

                dst_ptr = memory_obj.data_ptr

                # Cast the memoryview into a ctypes array type
                source_bfr = (ctypes.c_ubyte * obj_size).from_buffer(src_view)
                source_addr = ctypes.addressof(source_bfr)
                ctypes.memmove(dst_ptr, source_addr, obj_size)

                # Free shared memory buffer
                self.adhoc_shm_manager.free(recv, shm, mm)

                resource_idx += 1

        # Finally return the list of memory objects. At this point
        # some or all or none of them have the data retrieved from the bucket.
        return memory_objs

    def _get_objects_sync(
            self, keys: List[str],
            object_buffers: List[BufferGetObject]) -> bool:
        """Synchronous RDMA GET operation. This supports batched GETs too!"""

        # TODO: Need better handling of errors and partial failures

        try:
            client = self.client_pool[hash(keys[0]) % self.client_pool_size]

            # RDMA read into shared memory buffer
            client.get_object_buffers(object_buffers)

            return True
        except RuntimeError as e:
            # hpe_object raises RuntimeError for various errors
            error_str = str(e).lower()
            if ("not found" in error_str or
                "404" in error_str or
                "nosuchkey" in error_str):
                logger.debug("Object not found: %s", error_str)
                return False
            else:
                logger.error("RDMA GET error for %s: %s", keys[0], e)
                raise

        except Exception as e:
            logger.error("Unexpected error during RDMA GET %s: %s", keys[0], e)
            raise

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Put object to S3 using RDMA."""
        return await self.pq_executor.submit_job(
            self._put,
            key=key,
            memory_obj=memory_obj,
            priority=Priorities.PUT,
        )

    # Fire and forget version of _put()
    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Internal put implementation"""
        s3_key = self._make_s3_key(key)

        try:
            buffer_view = memory_obj.byte_array
            client = self.client_pool[hash(s3_key) % self.client_pool_size]

            put_buffer = BufferPutObject(
                bucket=self.settings.bucket,
                key=s3_key,
                buffer=buffer_view)

            # start_perf = time.perf_counter_ns()
            await self.loop.run_in_executor(
                None,
                client.put_object_buffers,
                put_buffer
            )

            # end_perf = time.perf_counter_ns()
            # perf_duration = end_perf - start_perf
            # logger.info(
            #     "%s RDMA PUT completed in %.6f ms: %s. Transfer size: %s",
            #     LOG_PREFIX, perf_duration / 1_000_000,
            #     s3_key,
            #     len(buffer_view))

            # Cache the size
            self.object_size_cache[s3_key] = len(buffer_view)

        except RuntimeError as e:
            logger.error("RDMA PUT error for %s: %s", s3_key, e)
            raise
        except Exception as e:
            logger.error("Unexpected error during RDMA PUT %s: %s", s3_key, e)
            raise

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> None:
        """Batched get implementation for RDMA"""
        return await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> None:
        """Internal implementation of batched put for RDMA"""

        buffer_objects: List[BufferPutObject] = []
        keys_list: List[str] = []

        for key, memory_obj in zip(keys, memory_objs):
            s3_key = self._make_s3_key(key)
            keys_list.append(s3_key)

            buffer_view = memory_obj.byte_array

            buffer_object = BufferPutObject(
                bucket=self.settings.bucket, key=s3_key, buffer=buffer_view)

            buffer_objects.append(buffer_object)

        try:
            client = \
                self.client_pool[hash(keys_list[0]) % self.client_pool_size]

            # start_perf = time.perf_counter_ns()

            await self.loop.run_in_executor(
                None,
                client.put_object_buffers,
                buffer_objects
            )

            # end_perf = time.perf_counter_ns()
            # perf_duration = end_perf - start_perf
            # logger.info(
            #     "%s RDMA PUT (batched) completed in %.6f ms for %d objects",
            #     LOG_PREFIX, perf_duration / 1_000_000,
            #     len(buffer_objects))
        except RuntimeError as e:
            logger.error("RDMA PUT error for %s: %s", keys_list[0], str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error during RDMA PUT %s: %s",
                         keys_list[0], e)
            raise

        # Cache the sizes for future reference
        for key, memory_obj in zip(keys_list, memory_objs):
            self.object_size_cache[key] = len(memory_obj.byte_array)

    # def support_batched_async_contains(self) -> bool:
        # return False

    def support_batched_get_non_blocking(self) -> bool:
        return False

    async def list(self) -> List[str]:
        """List all objects."""
        raise NotImplementedError

    def support_ping(self) -> bool:
        return False

    def support_batched_get(self) -> bool:
        return True

    def support_batched_put(self) -> bool:
        return True

    async def close(self) -> None:
        """Clean up resources."""
        logger.info("%s Closing S3 RDMA connector", LOG_PREFIX)

        if self.pq_executor:
            try:
                self.pq_executor.shutdown(wait=True)
            except Exception as e:
                logger.warning("Error shutting down executor: %s", e)

        # Clean up client pool
        self.client_pool.clear()

        self.object_size_cache.clear()
