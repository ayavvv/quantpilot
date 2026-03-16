"""Application entry point."""
from loguru import logger
from collector.scheduler import DataCollectorScheduler


def main():
    """Main function."""
    # Configure logging
    logger.add(
        "logs/collector_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="INFO",
        encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info("QuantPilot Collector starting")
    logger.info("=" * 60)

    # Create and start scheduler
    scheduler = DataCollectorScheduler()
    scheduler.start()


if __name__ == "__main__":
    main()
