import os

import sentry_sdk
from sentry_sdk.integrations.loguru import LoguruIntegration

from nos.version import __version__


NOS_TELEMETRY_ENABLED = bool(int(os.getenv("NOS_TELEMETRY_ENABLED", "0")))


if NOS_TELEMETRY_ENABLED:
    SENTRY_DSN = "https://e0fe7fff83449ac5163ddf62c7cc5011@o4504121578487808.ingest.sentry.io/4506114210004992"
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        debug=bool(int(os.getenv("SENTRY_DEBUG", "0"))),
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=0.2,
        release=__version__,
        integrations=[
            LoguruIntegration(),
        ],
    )
