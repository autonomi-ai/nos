import os

from nos.version import __version__


NOS_TELEMETRY_ENABLED = bool(int(os.getenv("NOS_TELEMETRY_ENABLED", "1")))


def _init_telemetry_logger() -> None:
    if NOS_TELEMETRY_ENABLED:
        import sentry_sdk  # noqa: I001
        from sentry_sdk.integrations.loguru import LoguruIntegration, LoggingLevels

        DEFAULT_SENTRY_DSN = (
            "https://e0fe7fff83449ac5163ddf62c7cc5011@o4504121578487808.ingest.sentry.io/4506114210004992"
        )
        SENTRY_DSN = os.getenv("SENTRY_DSN", DEFAULT_SENTRY_DSN)
        SENTRY_DEBUG = bool(int(os.getenv("SENTRY_DEBUG", "0")))
        # Loguru integration for Sentry
        sentry_loguru = LoguruIntegration(
            level=LoggingLevels.INFO.value,  # Capture info and above as breadcrumbs
            event_level=LoggingLevels.ERROR.value,  # Send errors as events
        )
        # Initialize Sentry
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            debug=SENTRY_DEBUG,
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            traces_sample_rate=1.0,
            # Set profiles_sample_rate to 1.0 to profile 100%
            # of sampled transactions.
            # We recommend adjusting this value in production.
            profiles_sample_rate=1.0,
            release=__version__,
            integrations=[
                sentry_loguru,
            ],
        )
