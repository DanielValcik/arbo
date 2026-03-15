"""Audit all data collection tables on VPS for backtesting readiness."""

import asyncio

from sqlalchemy import text

from arbo.utils.db import get_session_factory


async def main():
    factory = get_session_factory()
    async with factory() as session:
        # 1. Table row counts
        tables = [
            "weather_scan_log", "city_volume_daily", "health_checks",
            "paper_trades", "paper_positions", "paper_snapshots",
            "weather_forecasts", "signals", "markets", "daily_pnl",
            "real_market_data", "taker_flow_snapshots",
        ]
        print("=== TABLE ROW COUNTS ===")
        for t in tables:
            try:
                r = await session.execute(text(f"SELECT count(*) FROM {t}"))
                print(f"  {t}: {r.scalar()}")
            except Exception as e:
                print(f"  {t}: ERROR - {e}")

        # 2. weather_scan_log sample
        print("\n=== WEATHER_SCAN_LOG (last 5) ===")
        r = await session.execute(text(
            "SELECT city, target_date, edge, direction, market_price, forecast_prob, "
            "volume_24h, quality_gate_passed, quality_gate_reason, traded, trade_size "
            "FROM weather_scan_log ORDER BY scan_at DESC LIMIT 5"
        ))
        for row in r:
            print(f"  {row.city} {row.target_date} edge={row.edge:.3f} dir={row.direction} "
                  f"mkt_price={row.market_price:.4f} f_prob={row.forecast_prob:.4f} "
                  f"vol={row.volume_24h:.0f} passed={row.quality_gate_passed} "
                  f"traded={row.traded} size={row.trade_size}")
            if not row.quality_gate_passed:
                print(f"    reject reason: {row.quality_gate_reason}")

        # Scan log stats
        r = await session.execute(text(
            "SELECT quality_gate_passed, count(*) FROM weather_scan_log GROUP BY quality_gate_passed"
        ))
        print("  Pass/reject breakdown:", {str(row[0]): row[1] for row in r})

        r = await session.execute(text(
            "SELECT traded, count(*) FROM weather_scan_log GROUP BY traded"
        ))
        print("  Traded breakdown:", {str(row[0]): row[1] for row in r})

        # 3. city_volume_daily
        print("\n=== CITY_VOLUME_DAILY ===")
        r = await session.execute(text(
            "SELECT city, date, volume_24h, liquidity, num_markets, avg_price "
            "FROM city_volume_daily ORDER BY city, date"
        ))
        for row in r:
            avg_p = f" avg_p={row.avg_price:.4f}" if row.avg_price else ""
            print(f"  {row.city}: {row.date} vol=${float(row.volume_24h):.0f} "
                  f"liq=${float(row.liquidity):.0f} mkts={row.num_markets}{avg_p}")

        # 4. trade_details on paper_trades
        print("\n=== PAPER_TRADES.TRADE_DETAILS ===")
        r = await session.execute(text(
            "SELECT count(*) FROM paper_trades WHERE trade_details IS NOT NULL"
        ))
        with_details = r.scalar()
        r = await session.execute(text(
            "SELECT count(*) FROM paper_trades WHERE trade_details IS NULL AND strategy = 'C'"
        ))
        without_details = r.scalar()
        print(f"  With trade_details: {with_details}")
        print(f"  Strategy C WITHOUT trade_details: {without_details}")

        # Sample trade_details
        r = await session.execute(text(
            "SELECT id, strategy, trade_details FROM paper_trades "
            "WHERE trade_details IS NOT NULL ORDER BY placed_at DESC LIMIT 3"
        ))
        for row in r:
            td = row.trade_details or {}
            print(f"  #{row.id} strat={row.strategy} keys={list(td.keys())}")
            for k, v in td.items():
                print(f"    {k}: {v}")

        # 5. weather_forecasts
        print("\n=== WEATHER_FORECASTS ===")
        r = await session.execute(text("SELECT count(*) FROM weather_forecasts"))
        total = r.scalar()
        print(f"  Total: {total}")
        if total > 0:
            r = await session.execute(text(
                "SELECT city, forecast_date, source, temp_high_c, temp_low_c, "
                "actual_temp_high_c, fetched_at "
                "FROM weather_forecasts ORDER BY fetched_at DESC LIMIT 3"
            ))
            for row in r:
                print(f"  {row.city} {row.forecast_date} src={row.source} "
                      f"high={row.temp_high_c} low={row.temp_low_c} "
                      f"actual_high={row.actual_temp_high_c} fetched={row.fetched_at}")
        else:
            print("  WARNING: weather_forecasts is EMPTY — forecasts not being persisted!")

        # 6. signals table
        print("\n=== SIGNALS (last 3) ===")
        r = await session.execute(text(
            "SELECT id, layer, market_condition_id, direction, edge, "
            "confidence, details, detected_at "
            "FROM signals ORDER BY detected_at DESC LIMIT 3"
        ))
        for row in r:
            d = row.details or {}
            print(f"  #{row.id} layer={row.layer} dir={row.direction} "
                  f"edge={row.edge} conf={row.confidence} "
                  f"details_keys={list(d.keys())[:6]} at={row.detected_at}")

        # 7. Check data completeness for backtesting
        print("\n=== BACKTESTING READINESS ===")
        checks = []

        # Scan log captures rejected signals?
        r = await session.execute(text(
            "SELECT count(*) FROM weather_scan_log WHERE quality_gate_passed = false"
        ))
        rejected = r.scalar()
        checks.append(("Rejected signals captured", rejected > 0, f"{rejected} rows"))

        # Volume data per city
        r = await session.execute(text(
            "SELECT count(DISTINCT city) FROM city_volume_daily"
        ))
        cities = r.scalar()
        checks.append(("City volume tracking", cities >= 10, f"{cities} cities"))

        # Trade details on new trades
        checks.append(("Trade details (JSONB)", with_details > 0, f"{with_details} trades"))

        # Forecasts being saved
        checks.append(("Weather forecasts persisted", total > 0, f"{total} rows"))

        # Real market data snapshots
        r = await session.execute(text("SELECT count(*) FROM real_market_data"))
        rmd = r.scalar()
        checks.append(("Real market data snapshots", rmd > 0, f"{rmd} rows"))

        # Taker flow data
        r = await session.execute(text("SELECT count(*) FROM taker_flow_snapshots"))
        tfs = r.scalar()
        checks.append(("Taker flow snapshots", tfs > 0, f"{tfs} rows"))

        for name, ok, detail in checks:
            status = "OK" if ok else "MISSING"
            print(f"  [{status}] {name}: {detail}")


asyncio.run(main())
