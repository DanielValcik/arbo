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

    service = await asyncio.get_event_loop().run_in_executor(None, _get_service)
    if service is None:
        return {"status": "no_service", "redeemed": 0}

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: service.redeem_all(batch_size=10)
        )

        redeemed = len(result.success_list)
        errors = len(result.error_list)

        if redeemed > 0:
            logger.info(
                "auto_redeem_success",
                redeemed=redeemed,
                errors=errors,
            )
        elif errors > 0:
            logger.warning("auto_redeem_errors", errors=errors)

        return {
            "status": "ok",
            "redeemed": redeemed,
            "errors": errors,
            "tx": result.success_list[0].get("transactionHash", "") if result.success_list else "",
        }

    except Exception as e:
        err_msg = str(e)
        if "No redeemable positions" in err_msg or "nothing to redeem" in err_msg.lower():
            return {"status": "nothing_to_redeem", "redeemed": 0}
        logger.warning("auto_redeem_failed", error=err_msg)
        return {"status": "error", "error": err_msg, "redeemed": 0}
