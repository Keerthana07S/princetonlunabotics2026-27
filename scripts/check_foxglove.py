#!/usr/bin/env python3
"""Connect to foxglove_bridge like the Foxglove app and count messages per topic.

    python3 scripts/check_foxglove.py --seconds 10 /viz/map /camera/left/image_raw

Prints JSON with the advertised topics and the messages received on each
requested topic. Exits 1 if the bridge is unreachable or a requested topic
received nothing.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time

from websockets.sync.client import connect

MESSAGE_DATA = 0x01


def check(url: str, topics: list[str], seconds: float) -> dict:
    counts = {topic: 0 for topic in topics}
    advertised: dict[int, str] = {}
    subscribed: dict[int, str] = {}
    # foxglove_bridge 3.x speaks foxglove.sdk.v1; older bridges foxglove.websocket.v1.
    with connect(url, subprotocols=["foxglove.sdk.v1", "foxglove.websocket.v1"], max_size=None, open_timeout=10) as socket:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                frame = socket.recv(timeout=max(deadline - time.monotonic(), 0.01))
            except TimeoutError:
                break
            if isinstance(frame, str):
                message = json.loads(frame)
                if message.get("op") != "advertise":
                    continue
                for channel in message["channels"]:
                    advertised[int(channel["id"])] = channel["topic"]
                    if channel["topic"] in counts and channel["topic"] not in subscribed.values():
                        subscription = len(subscribed) + 1
                        subscribed[subscription] = channel["topic"]
                        socket.send(
                            json.dumps({"op": "subscribe", "subscriptions": [{"id": subscription, "channelId": channel["id"]}]})
                        )
            elif frame and frame[0] == MESSAGE_DATA:
                (subscription,) = struct.unpack_from("<I", frame, 1)
                topic = subscribed.get(subscription)
                if topic is not None:
                    counts[topic] += 1
    return {"url": url, "advertised": sorted(set(advertised.values())), "received": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topics", nargs="*", default=["/viz/map", "/viz/ground_truth_path"])
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    try:
        result = check(args.url, args.topics, args.seconds)
    except OSError as exc:
        print(json.dumps({"url": args.url, "error": str(exc)}))
        sys.exit(1)
    print(json.dumps(result, indent=2))
    sys.exit(0 if all(result["received"].values()) else 1)


if __name__ == "__main__":
    main()
