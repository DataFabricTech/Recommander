from app.routers import routers
from app.server import *
from common.log import *


from apscheduler.schedulers.background import BackgroundScheduler


def include_routers():
    for router in routers:
        app.include_router(router)

    from fastapi.middleware.cors import CORSMiddleware
    _all = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_all,
        allow_credentials=True,
        allow_methods=_all,
        allow_headers=_all,
    )

    return app


def start():
    import uvicorn

    if Config.sentry.enable:
        from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
        app.add_middleware(SentryAsgiMiddleware)

    uvicorn.run(
        "main:include_routers",
        host=Config.server.host,
        port=Config.server.port,
        log_config=log_config,
        **Config.server.uvicorn
    )


if __name__ == '__main__':
    from app.routers.training_service import init_clustering
    from app.routers.training_service import init_embedding

    if Config.sentry.enable:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=Config.sentry.dsn,
            integrations=[
                LoggingIntegration(
                    level=Config.sentry.level,
                    event_level=Config.sentry.event_level,
                ),
            ],
            traces_sample_rate=Config.sentry.traces_sample_rate,
            debug=Config.sentry.debug,
            environment=Config.sentry.environment,
        )
    scheduler = BackgroundScheduler()

    scheduler.add_job(init_clustering, 'cron', hour=Config.cron.hour, minute=Config.cron.minute)
    scheduler.add_job(init_embedding, 'cron', hour=Config.cron.hour, minute=Config.cron.minute)

    start()
