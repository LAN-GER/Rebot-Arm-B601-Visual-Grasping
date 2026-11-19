"""Latest-frame-only preview worker, isolated from capture and robot execution."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from rebot_grasp.core.types import Capture


@dataclass(frozen=True)
class PreviewFrame:
    capture: Capture
    backend_view: object
    result: object
    image: object
    duration_s: float


class LivePreview:
    def __init__(self, thread_name):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self.future = None
        self.latest = None
        self.closed = False

    def _process(self, frame):
        raise NotImplementedError

    def tick(self, frame):
        """Return a newly completed result, then submit this frame if idle.

        Busy frames are dropped, not queued. GUI and camera stay on their owner
        thread; only detection, point-cloud processing and image drawing run here.
        """
        if self.closed:
            return None
        ready = None
        if self.future is not None and self.future.done():
            ready = self.future.result()
            self.latest = ready
            self.future = None
        if self.future is None:
            self.future = self.executor.submit(self._process, frame)
        return ready

    def pause(self):
        """Drain shared inference before an explicit G snapshot; never reuse it."""
        if self.future is not None:
            try:
                self.future.result()
            finally:
                self.future = None
        self.latest = None

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = None
        self.latest = None
