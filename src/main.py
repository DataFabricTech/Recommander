from app.routers import routers
from app.server import *
from common.log import *

from app.routers.training_service import recommendation_data_model

from apscheduler.schedulers.background import BackgroundScheduler

'''
1. '신규 데이터'에 대한 기존 데이터 추천
2. '주기적인' 기존 데이터에 대한 벡터화
'''


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

    scheduler.add_job(recommendation_data_model, 'cron', hour=Config.cron.hour, minute=Config.cron.minute)

    start()
