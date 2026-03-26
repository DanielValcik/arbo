"""Check C2 positions vs autoresearch model expectations."""
import asyncio
from arbo.utils.db import PaperTrade, get_session_factory
from sqlalchemy import select

# Autoresearch winning params
MODEL = {
    "min_edge": 0.03,
    "max_price": 0.45,
    "min_price": 0.03,
    "min_hold_edge": 0.05,
    "profit_target_abs": 0.15,
    "prob_exit_floor": 0.10,
    "kelly_raw_cap": 0.30,
    "prob_sharpening": 0.85,
    "excluded": {"sao_paulo", "tel_aviv", "tokyo", "lucknow"},
}

async def main():
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PaperTrade)
            .where(PaperTrade.strategy == "C2")
            .order_by(PaperTrade.placed_at)
        )
        trades = result.scalars().all()

        print(f"C2 Model: EMOS-Exit-Fusion (score=138.1)")
        print(f"Exit logic: sell when edge < {MODEL['min_hold_edge']}, "
              f"profit take at +${MODEL['profit_target_abs']}, "
              f"prob floor at {MODEL['prob_exit_floor']}")
        print(f"Trades: {len(trades)}")
        print()

        issues = []
        for r in trades:
            td = r.trade_details or {}
            city = td.get("city", "?")
            direction = td.get("direction", "?")
            edge = float(r.edge_at_exec or 0)
            gamma_price = td.get("market_price_gamma", 0)
            clob_fill = td.get("clob_fill_p1")
            forecast_prob = td.get("forecast_prob", 0)
            target_date = td.get("target_date", "?")

            # Determine effective price for quality gate check
            if direction == "BUY_YES":
                eff_price = gamma_price
            else:
                eff_price = 1.0 - gamma_price if gamma_price else 0

            # Profit take trigger price
            entry_for_exit = float(clob_fill) if clob_fill else eff_price
            profit_trigger = entry_for_exit + MODEL["profit_target_abs"]

            print(f"{'='*60}")
            print(f"{city} {direction}  target_date={target_date}  status={r.status}")
            print(f"  Entry: gamma_yes={gamma_price}, clob_fill={clob_fill}, eff_price={eff_price:.4f}")
            print(f"  Edge: {edge:.4f} (min={MODEL['min_edge']})")
            print(f"  Forecast prob: {forecast_prob}")
            print(f"  Size: ${float(r.size):.2f}")
            print(f"  Placed: {r.placed_at}")

            # Validate against model
            checks = []
            if city in MODEL["excluded"]:
                checks.append(f"ISSUE: {city} is in excluded cities!")
                issues.append(f"{city}: should be excluded")

            if edge < MODEL["min_edge"]:
                checks.append(f"ISSUE: edge {edge:.4f} < min_edge {MODEL['min_edge']}")
                issues.append(f"{city}: edge too low")

            if eff_price > MODEL["max_price"]:
                checks.append(f"ISSUE: eff_price {eff_price:.4f} > max_price {MODEL['max_price']}")
                issues.append(f"{city}: price too high")

            if eff_price < MODEL["min_price"]:
                checks.append(f"ISSUE: eff_price {eff_price:.4f} < min_price {MODEL['min_price']}")
                issues.append(f"{city}: price too low")

            if not checks:
                checks.append("OK — passes quality gate")

            for c in checks:
                print(f"  {c}")

            # Exit conditions
            print(f"\n  Exit triggers:")
            print(f"    Profit take: price >= {profit_trigger:.4f} (entry {entry_for_exit:.4f} + ${MODEL['profit_target_abs']})")
            print(f"    Edge lost: updated_edge < {MODEL['min_hold_edge']} (currently {edge:.4f})")
            print(f"    Prob floor: price < {MODEL['prob_exit_floor']}")
            print(f"    Resolution: METAR actual temp on {target_date} (if no exit triggers first)")

        if issues:
            print(f"\n{'='*60}")
            print(f"ISSUES FOUND: {len(issues)}")
            for i in issues:
                print(f"  - {i}")
        else:
            print(f"\n{'='*60}")
            print(f"All {len(trades)} trades pass model validation.")

asyncio.run(main())
