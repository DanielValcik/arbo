"""Auto-redeem resolved Polymarket positions via gasless relayer.

Uses poly-web3 + Builder API to redeem winning positions through
Polymarket's gasless relayer. No MATIC needed.

Runs as periodic task (every 30 min). Redeems all resolved positions
in a single batch transaction.
"""

from __future__ import annotations

import os

from arbo.utils.logger import get_logger

logger = get_logger("auto_redeem")

_service = None


def _get_service():
    """Lazy-init PolyWeb3Service (heavy imports)."""
    global _service
    if _service is not None:
        return _service

    try:
        from poly_web3 import PolyWeb3Service
        from py_builder_relayer_client.client import RelayClient
        from py_builder_signing_sdk.config import BuilderConfig
        from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
        from py_clob_client.client import ClobClient

        pk = os.environ.get("POLY_PRIVATE_KEY", "")
        funder = os.environ.get("POLY_FUNDER_ADDRESS", "")
        builder_key = os.environ.get("BUILDER_KEY", "")
        builder_secret = os.environ.get("BUILDER_SECRET", "")
        builder_pass = os.environ.get("BUILDER_PASSPHRASE", "")

        if not all([pk, funder, builder_key, builder_secret, builder_pass]):
            logger.warning("auto_redeem_missing_creds")
            return None

        clob = ClobClient(
            "https://clob.polymarket.com",
            key=pk,
            chain_id=137,
            signature_type=2,
            funder=funder,
        )
        clob.set_api_creds(clob.create_or_derive_api_creds())

        relayer = RelayClient(
            "https://relayer-v2.polymarket.com/",
            137,
            pk,
            BuilderConfig(local_builder_creds=BuilderApiKeyCreds(
                key=builder_key,
                secret=builder_secret,
                passphrase=builder_pass,
            )),
        )

        _service = PolyWeb3Service(
            clob_client=clob,
            relayer_client=relayer,
            rpc_url="https://polygon-bor-rpc.publicnode.com",
        )
        logger.info("auto_redeem_service_ready")
        return _service

    except Exception as e:
        logger.warning("auto_redeem_init_failed", error=str(e))
        return None


async def redeem_resolved_positions() -> dict:
    """Redeem all resolved positions. Returns summary dict."""
    import asyncio
    import json as _json
    import ssl
    import urllib.request

    service = await asyncio.get_event_loop().run_in_executor(None, _get_service)
    if service is None:
        return {"status": "no_service", "redeemed": 0}

    loop = asyncio.get_event_loop()

    # Step 1: Find redeemable positions (value > 0) from Data API
    funder = os.environ.get("POLY_FUNDER_ADDRESS", "")
    try:
        _ssl = ssl.create_default_context()
        try:
            import certifi
            _ssl = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        url = f"https://data-api.polymarket.com/positions?user={funder}"
        req = urllib.request.Request(url, headers={"User-Agent": "Arbo/1.0"})
        data = await loop.run_in_executor(None, lambda: _json.loads(
            urllib.request.urlopen(req, timeout=10, context=_ssl).read()
        ))
        redeemable_ids = [
            p["conditionId"] for p in data
            if float(p.get("currentValue", 0)) > 0
            and p.get("conditionId")
            and p.get("redeemable") is True
        ]
        logger.info(
            "auto_redeem_scan",
            total_positions=len(data),
            with_value=sum(1 for p in data if float(p.get("currentValue", 0)) > 0),
            redeemable=len(redeemable_ids),
        )
    except Exception as e:
        logger.warning("auto_redeem_positions_check_failed", error=str(e))
        return {"status": "error", "error": str(e), "redeemed": 0}

    if not redeemable_ids:
        return {"status": "nothing_to_redeem", "redeemed": 0}

    # Step 2: Redeem positions one by one (batch can fail on mixed states)
    try:
        logger.info("auto_redeem_attempting", count=len(redeemable_ids))
        # Redeem each condition individually to avoid batch errors
        result = await loop.run_in_executor(
            None, lambda: service.redeem(redeemable_ids, batch_size=1)
        )

        redeemed = len(result.success_list)
        errors = len(result.error_list)

        if redeemed > 0:
            logger.info("auto_redeem_success", redeemed=redeemed, errors=errors)
        if errors > 0:
            err_ids = getattr(result, "error_condition_ids", [])
            logger.warning(
                "auto_redeem_errors", errors=errors,
                error_ids=[e[:16] for e in err_ids[:3]],
                error_list=[str(e)[:100] for e in result.error_list[:3]],
            )

        return {
            "status": "ok",
            "redeemed": redeemed,
            "errors": errors,
            "redeemed_condition_ids": redeemable_ids[:redeemed],
        }

    except Exception as e:
        err_msg = str(e)
        logger.warning("auto_redeem_failed", error=err_msg)
        return {"status": "error", "error": err_msg, "redeemed": 0}
