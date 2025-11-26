# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from functools import partial
from itertools import cycle
#from threading import Lock
from typing import List, Optional
from urllib.parse import quote as url_quote
import asyncio
import ctypes
import mmap
import os
import tempfile
import time

# Third Party
from awscrt import auth, io, s3
from awscrt.http import HttpHeaders, HttpRequest
from awscrt.io import ClientTlsContext, TlsConnectionOptions, TlsContextOptions

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

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# Unique prefix for easy log filtering
LOG_PREFIX = "[S3-TCP]"


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


# TODO(Jiayi): Some pending problems.
# (1) We might need a filesystem-like allocator.
# This could be useful for local disk `LocalDiskBackend` and
# `/dev/shm` in `S3Connector`
# (2) Need to hack amazon python s3 crt library to enable `offset`
# to achieve zero-copy.
# (3) Need a job manager so that we can do sth like
# write priority, read priority, etc.
# (4) Potentially can drop the semaphore to reduce the complexity.
# Let crt handle the scheduling.


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


class S3Connector(RemoteConnector):
    """
    S3 remote connector
    """

    def __init__(
        self,
        s3_endpoint: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        s3_part_size: Optional[int],
        s3_file_prefix: Optional[str],
        s3_max_io_concurrency: int,
        s3_max_inflight_reqs: int,
        s3_prefer_http2: bool,
        s3_region: str,
        s3_enable_s3express: bool,
    ):
        if not s3_endpoint.startswith("s3://"):
            raise ValueError("S3 url must start with 's3://'")

        self.s3_endpoint = s3_endpoint.removeprefix("s3://")
        self.s3_bucket_name = "s3tcpbucket"
        self.s3_rdma_endpoint = "http://cxo-s3-cluster200.lab.nimblestorage.com:8443"
        self.s3_prefix = s3_file_prefix
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.s3_part_size = s3_part_size

        # TODO(Jiayi): Now we only assume S3 part size = chunk size
        assert self.s3_part_size == self.full_chunk_size, (
            "S3 part size must be equal to chunk size in S3Connector"
        )

        self.s3_max_io_concurrency = s3_max_io_concurrency
        self.s3_max_inflight_reqs = s3_max_inflight_reqs
        self.s3_prefer_http2 = s3_prefer_http2
        self.s3_region = s3_region
        self.s3_enable_s3express = s3_enable_s3express

        event_loop_group = io.EventLoopGroup(s3_max_io_concurrency)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        self.credentials_provider = auth.AwsCredentialsProvider.new_default_chain(
            client_bootstrap
        )

        tls_opts = None
        if self.s3_prefer_http2:
            # Use HTTP/2 multiplexing if possible.
            tls_ctx = ClientTlsContext(TlsContextOptions())
            tls_opts = TlsConnectionOptions(tls_ctx)
            try:
                tls_opts.set_alpn_list(["h2", "http/1.1"])
            except Exception:
                tls_opts = None

        logger.info("Initializing S3 client")
        self.s3_client = s3.S3Client(
            bootstrap=client_bootstrap,
            region=s3_region,
            credential_provider=self.credentials_provider,
            enable_s3express=False,  # enable for s3express
            tls_connection_options=tls_opts,
            tls_mode=s3.S3RequestTlsMode.DISABLED,  # only for non-AWS services
        )

        # TODO(Jiayi): We need to handle cache consistency issues in a systematic way
        # across all connectors.
        # We assume S3 cache is never evicted and read-only for now.
        # the object size cache does not need protection because
        # asyncio scheduling is cooperative and not preemptive
        self.object_size_cache: dict[str, int] = {}

        self.inflight_sema = asyncio.Semaphore(s3_max_inflight_reqs)
        self.pq_executor = AsyncPQExecutor(loop)

        self._client_pool: List[S3RdmaClient] = []
        self._client_pool_size = 16 # Number of clients in the pool
        self._client_iterator: Optional[cycle] = None
        self._client_lock = asyncio.Lock()


    def post_init(self):
        logger.info("Post-initializing S3 connector")

        if self.s3_part_size is None:
            # Default to chunk size
            self.s3_part_size = self.full_chunk_size
        assert self.s3_part_size == self.full_chunk_size, (
            "S3 part size must be equal to chunk size in S3Connector"
        )

        shm_name_prefix = "my_shm"
        shms = []
        shm_names = []
        mmaps = []
        for i in range(self.s3_max_inflight_reqs):
            shm_name = f"{shm_name_prefix}_{i}"

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
            endpoint=self.s3_rdma_endpoint,
            max_parallel_requests=16,
        )

        # if self.setings.max_segment_size is not None:
        #     logger.info("Using max segment size: %s bytes", self.settings.max_segment_size)
        #     client_config.max_segment_size = self.settings.max_segment_size

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

    # Pick the next S3RdmaClient from the pool
    async def _get_next_client(self) -> S3RdmaClient:
        """Get next client from pool using round-robin scheduling."""
        assert self._client_iterator is not None

        async with self._client_lock:
            return next(self._client_iterator)

    def _format_safe_path(self, key_str: str) -> str:
        """
        Generate a safe HTTP path for the S3 key.
        This is necessary because S3 keys can contain special characters
        that need to be URL-encoded.
        """
        flat_key_str = key_str.replace("/", "_")
        if self.s3_prefix:
            path = f"/{self.s3_prefix}/{flat_key_str}"
        else:
            path = f"/{flat_key_str}"
        # Keep slashes as they are path separators in S3.
        return url_quote(path, safe="/")

    # TODO(Jiayi): optimize this with async
    def _get_object_size(self, key_str: str) -> int:
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        req = HttpRequest("HEAD", self._format_safe_path(key_str), headers)

        got = {"len": None, "status": None, "err": None}

        def on_headers(status_code, headers, **kwargs):
            got["status"] = status_code
            for name, value in headers:
                if name.lower() == "content-length":
                    try:
                        got["len"] = int(value)
                    except Exception:
                        pass

        def on_done(error=None, **kwargs):
            got["err"] = error

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="HeadObject",
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
        )

        try:
            s3_req.finished_future.result()
        except Exception as e:
            logger.debug(f"Exception in `_get_object_size`: {e}")
            return 0
        if got["err"] or got["status"] != 200:
            logger.warning(
                "Encountering error in S3 HEAD request "
                f"with error code: {got['status']}"
            )
            return 0
        return got["len"] if got["len"] is not None else 0

    # exactly the same as _get_object_size just awaiting an asyncio.Future
    # instead of a concurrent.futures.Future
    async def _get_object_size_async(self, key_str: str) -> int:
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        req = HttpRequest("HEAD", self._format_safe_path(key_str), headers)

        got = {"len": None, "status": None, "err": None}

        def on_headers(status_code, headers, **kwargs):
            got["status"] = status_code
            for name, value in headers:
                if name.lower() == "content-length":
                    try:
                        got["len"] = int(value)
                    except Exception:
                        pass

        def on_done(error=None, **kwargs):
            got["err"] = error

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="HeadObject",
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
        )

        try:
            await asyncio.wrap_future(s3_req.finished_future)
        except Exception as e:
            logger.debug(f"Exception in `_get_object_size_async`: {e}")
            return 0
        if got["err"] or got["status"] != 200:
            logger.warning(
                "Encountering error in S3 HEAD request "
                f"with error code: {got['status']}"
            )
            return 0
        return got["len"] if got["len"] is not None else 0

    # TODO(Jiayi): implement real async
    async def exists(self, key: CacheEngineKey) -> bool:
        return self.exists_sync(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        key_str = key.to_string()
        if key_str in self.object_size_cache:
            return self.object_size_cache[key_str] > 0
        cache_size = self._get_object_size(key_str)
        if cache_size > 0:
            self.object_size_cache[key_str] = cache_size
            return True
        return False

    def _s3_download(
        self,
        key_str: str,
        recv_path: str,
    ):
        """
        Download a file from S3.
        """
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)

        # TODO(Jiayi): Enable more finegrained data partition
        # range_header = f"bytes={start_byte}-{end_byte}"
        # headers.add("Range", range_header)

        req = HttpRequest("GET", self._format_safe_path(key_str), headers)

        # NOTE(Jiayi): Run in crt threads (not this thread) with GIL
        # See https://github.com/awslabs/aws-crt-python/blob/4250709624119de1af3ca86816e1a154fcac7cc8/source/common.c#L51
        def on_done(error=None, status_code=None, **kwargs):
            ok = (status_code in (200, 206)) or (status_code is None)
            if error or not ok:
                raise RuntimeError(
                    f"Failed to download {key_str} from S3: {error or status_code}"
                )

        # TODO(Jiayi): Need to support offset to enable zero-copy
        # More concretely, we need to get the shared memory offset.
        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.GET_OBJECT,
            request=req,
            operation_name="GetObject",
            recv_filepath=recv_path,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )

        return s3_req

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        obj_size = self.object_size_cache.get(key_str, None)

        if obj_size is None:
            obj_size = await self._get_object_size_async(key_str)
            if obj_size <= 0:
                self.object_size_cache[key_str] = 0
                return None
            self.object_size_cache[key_str] = obj_size

        #await self.inflight_sema.acquire()

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shape,
            self.meta_dtype,
            self.meta_fmt,
        )

        # TODO(Jiayi): Please support this
        assert obj_size == memory_obj.get_size(), (
            "Saving unfull chunk is not supported in S3Connector."
        )

        # TODO(Jiayi): Need to support offset to enable zero-copy
        # We probably need to get the shared memory offset directly from memory object.
        recv_path, shm, mm = self.adhoc_shm_manager.allocate()
        src_view = memoryview(mm)

        #start_perf = time.perf_counter_ns()
        # s3_req = self._s3_download(
        #     key_str=key_str,
        #     recv_path=recv_path,
        # )
        # await asyncio.wrap_future(s3_req.finished_future)
        #next_client = await self._get_next_client()
        #assert next_client is not None

        # Lockless client selection using consistent hashing
        next_client = self._client_pool[hash(key_str) % self._client_pool_size]

        # The keys are stored with '/' replaced by '_'
        # We'll need the same format while retrieving
        flat_key_str = key_str.replace("/", "_")

        buffer_get_object_instance = BufferGetObject(
            bucket=self.s3_bucket_name,
            key=flat_key_str,
            buffer=src_view
        )

        # Wrap synchronous RDMA call in executor to properly await it
        await self.loop.run_in_executor(
            None,
            next_client.get_object_buffers,
            buffer_get_object_instance
        )

        dst_ptr = memory_obj.data_ptr

        # Cast the memoryview into a ctypes array type (e.g., an array of bytes)
        # This makes it compatible with ctypes' internal pointer logic
        c_source_buffer = (ctypes.c_ubyte * obj_size).from_buffer(src_view)

        # Get the address of that ctypes buffer object
        source_address = ctypes.addressof(c_source_buffer)
        ctypes.memmove(dst_ptr, source_address, obj_size)

        #end_perf = time.perf_counter_ns()
        #perf_duration = end_perf - start_perf
        # logger.info(
        #     "%s TCP GET completed in %.6f ms: %s. Transfer size: %s",
        #     LOG_PREFIX, perf_duration / 1_000_000, key_str, obj_size)

        self.adhoc_shm_manager.free(recv_path, shm, mm)

        #self.inflight_sema.release()

        return memory_obj

    # this callback allows us to safely have multiple calls to batched_get
    # since we release the semaphores 1-by-1
    def on_get_done(
        self,
        obj_size: int,
        memory_obj: MemoryObj,
        shm: int,
        mm: mmap.mmap,
        recv_path: str,
        #key_str: str,
        #start_time: int,
        #fut: asyncio.Future,
    ):
        try:
            if memory_obj is None or shm is None:
                return None

            dst_ptr = memory_obj.data_ptr
            ctypes.memmove(dst_ptr, shm, obj_size)

            self.adhoc_shm_manager.free(recv_path, shm, mm)

            # _end = time.perf_counter_ns()
            # _duration_ms = _end - start_time
            # logger.info(
            #     "%s TCP GET completed in %.6f ms: %s. Transfer size: %s",
            #     LOG_PREFIX, _duration_ms / 1_000_000, key_str, obj_size)

        except Exception as e:
            logger.error("on_get_done failed for %s : %s", recv_path, str(e))
        finally:
            self.inflight_sema.release()

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        memory_objs: List[Optional[MemoryObj]] = []
        buffer_get_objects: List[BufferGetObject] = []
        resources: List[tuple[str, int, mmap.mmap, memoryview]] = []

        # It is okay for len(keys) > self.s3_max_inflight_reqs
        # but it will be slower.
        if len(keys) > self.s3_max_inflight_reqs:
            logger.warning(
                f"More keys {len(keys)} to get than "
                f"max inflight requests {self.s3_max_inflight_reqs}."
                "This will cause slower retrieval."
            )

        # Prepare all buffer objects and allocate resources
        for key in keys:
            key_str = key.to_string()

            obj_size = self.object_size_cache.get(key_str, None)

            if obj_size is None:
                obj_size = await self._get_object_size_async(key_str)
                if obj_size <= 0:
                    self.object_size_cache[key_str] = 0
                    memory_objs.append(None)
                    continue
                self.object_size_cache[key_str] = obj_size

            memory_obj = self.local_cpu_backend.allocate(
                self.meta_shape,
                self.meta_dtype,
                self.meta_fmt,
            )

            memory_objs.append(memory_obj)

            if not memory_obj:
                continue

            # TODO(Jiayi): Please support this
            assert obj_size == memory_obj.get_size(), (
                "Saving unfull chunk is not supported in S3Connector."
            )

            # Allocate shared memory buffer
            recv_path, shm, mm = self.adhoc_shm_manager.allocate()
            src_view = memoryview(mm)

            # Store resources for cleanup later
            resources.append((recv_path, shm, mm, src_view))

            # The keys are stored with '/' replaced by '_'
            flat_key_str = key_str.replace("/", "_")

            buffer_get_object_instance = BufferGetObject(
                bucket=self.s3_bucket_name,
                key=flat_key_str,
                buffer=src_view
            )

            buffer_get_objects.append(buffer_get_object_instance)

        # Perform batched RDMA read if we have any valid objects
        if buffer_get_objects:
            # Select client using consistent hashing based on first key
            next_client = self._client_pool[hash(keys[0].to_string()) % self._client_pool_size]

            logger.info(
                "%s Starting RDMA Batched GET", LOG_PREFIX)

            start_perf = time.perf_counter_ns()
            # Wrap synchronous RDMA call in executor to properly await it
            await self.loop.run_in_executor(
                None,
                next_client.get_object_buffers,
                buffer_get_objects
            )
            perf_duration = time.perf_counter_ns() - start_perf
            logger.info(
                "%s RDMA Batched GET completed in %.6f ms: %d objects.",
                LOG_PREFIX, perf_duration / 1_000_000,
                len(buffer_get_objects)
            )

            # Copy data from shared memory buffers to memory objects
            resource_idx = 0
            for i, memory_obj in enumerate(memory_objs):
                if memory_obj is None:
                    continue

                recv_path, shm, mm, src_view = resources[resource_idx]
                key_str = keys[i].to_string()
                obj_size = self.object_size_cache[key_str]

                dst_ptr = memory_obj.data_ptr

                # Cast the memoryview into a ctypes array type
                c_source_buffer = (ctypes.c_ubyte * obj_size).from_buffer(src_view)
                source_address = ctypes.addressof(c_source_buffer)
                ctypes.memmove(dst_ptr, source_address, obj_size)

                # Free shared memory buffer
                self.adhoc_shm_manager.free(recv_path, shm, mm)

                resource_idx += 1

        return memory_objs

    def _s3_upload(
        self,
        key_str: str,
        send_path: str,
    ):
        """
        Upload a file to S3.
        """
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)

        req = HttpRequest("PUT", self._format_safe_path(key_str), headers)

        done = {"err": None, "status": None}

        def on_done(error=None, status_code=None, **kwargs):
            done["err"] = error
            done["status"] = status_code

            if done["err"] or done["status"] not in (200, 201):
                raise RuntimeError(f"Upload failed in S3Connector: {done}")

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.PUT_OBJECT,
            request=req,
            operation_name="PutObject",
            send_filepath=send_path,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )
        return s3_req

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Store data to S3
        """

        key_str = key.to_string()

        # TODO(Jiayi): Please support this
        assert memory_obj.get_physical_size() == self.s3_part_size, (
            "Saving unfull chunk is not supported in S3Connector."
        )

        await self.inflight_sema.acquire()
        send_path, shm, mm = self.adhoc_shm_manager.allocate()
        logger.debug("Allocated shared memory for S3 upload")

        try:
            buffer_ptr = memory_obj.data_ptr
            ctypes.memmove(shm, buffer_ptr, memory_obj.get_physical_size())
            logger.debug("Data copy to S3 buffer completed")

            # _start = time.perf_counter_ns()
            s3_req = self._s3_upload(key_str, send_path)
            await asyncio.wrap_future(s3_req.finished_future)
            # _end = time.perf_counter_ns()
            # _duration_ms = (_end - _start)
            # logger.info("%s TCP PUT completed in %.6f ms: %s. Transfer size: %s", LOG_PREFIX, _duration_ms / 1_000_000, key_str, memory_obj.get_physical_size())

            self.object_size_cache[key_str] = memory_obj.get_physical_size()
            logger.debug(f"Uploaded {key_str} to S3 successfully")
        except Exception as e:
            logger.error(f"Failed to upload {key_str} to S3: {e}")
            raise
        finally:
            self.inflight_sema.release()
            self.adhoc_shm_manager.free(send_path, shm, mm)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        return await self.pq_executor.submit_job(
            self._put,
            key=key,
            memory_obj=memory_obj,
            priority=Priorities.PUT,
        )

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            key_str = key.to_string()
            cached_size = self.object_size_cache.get(key_str, None)
            if cached_size is not None:
                if cached_size > 0:
                    num_hit_counts += 1
                    continue
                else:
                    return num_hit_counts

            obj_size = await self._get_object_size_async(key_str)
            if not obj_size > 0:
                self.object_size_cache[key_str] = 0
                return num_hit_counts

            self.object_size_cache[key_str] = obj_size
            num_hit_counts += 1

        return num_hit_counts

    async def batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # batched get is already a coroutine
        result = await self.batched_get(keys)
        return [r for r in result if r is not None]

    async def batched_get_non_blocking(
        self, lookup_id: str, keys: List[CacheEngineKey]
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    async def list(self) -> List[str]:
        raise NotImplementedError

    def support_ping(self) -> bool:
        return False

    # TODO(Jiayi): This needs to be implemented.
    async def ping(self) -> int:
        raise NotImplementedError

    def support_batched_get(self) -> bool:
        return True

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
        self.adhoc_shm_manager.close()
