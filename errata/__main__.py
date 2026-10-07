from __future__ import annotations

import uvicorn

from .config import load_config, setup_logging
from .webapp import create_app


def main() -> None:
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    app = create_app(config)
    server = config["server"]
    uvicorn.run(
        app,
        host=server["listen_host"],
        port=int(server["listen_port"]),
        log_config=None,
    )


if __name__ == "__main__":
    main()
