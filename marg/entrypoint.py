import os
import threading

import uvicorn

from marg.api.app import app
from marg.worker import worker_loop


def main() -> None:
    worker_thread = None
    if os.environ.get("MARG_SQS_QUEUE_URL") and os.environ.get("MARG_S3_BUCKET"):
        worker_thread = threading.Thread(target=worker_loop, daemon=True, name="marg-worker")
        worker_thread.start()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
    if worker_thread is not None:
        worker_thread.join(timeout=1)


if __name__ == "__main__":
    main()
