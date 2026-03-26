"""Check transaction status on Polygon."""
import os
from dotenv import load_dotenv
load_dotenv()
from web3 import Web3

rpc = os.getenv("ALCHEMY_POLYGON_URL") or "https://rpc.ankr.com/polygon"
w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))

tx_hash = "0xa6a1ad5274f0f69ecbab560c05db3fe14154fbfe572021d22aad6139b4e37303"
funder = Web3.to_checksum_address(os.getenv("POLY_FUNDER_ADDRESS"))

try:
    tx = w3.eth.get_transaction(tx_hash)
    print("FOUND ON POLYGON")
    print("From:", tx["from"])
    print("To:", tx["to"])
    print("Value:", tx["value"] / 1e18, "POL")
    print("Block:", tx["blockNumber"])

    to_match = str(tx["to"]).lower() == funder.lower()
    print("To matches funder:", to_match)

    receipt = w3.eth.get_transaction_receipt(tx_hash)
    print("Status:", "SUCCESS" if receipt["status"] == 1 else "FAILED")
    print("Gas used:", receipt["gasUsed"])

    # Check if it's a token transfer (not native POL)
    if receipt["logs"]:
        print(f"\nLogs ({len(receipt['logs'])} events):")
        for log in receipt["logs"][:5]:
            print("  Contract:", log["address"])
            print("  Topics:", [t.hex()[:20] + "..." for t in log["topics"]])

except Exception as e:
    print(f"Not found on Polygon: {e}")

# Also check current funder balance
pol = w3.eth.get_balance(funder) / 1e18
print(f"\nFunder POL balance now: {pol:.10f}")
