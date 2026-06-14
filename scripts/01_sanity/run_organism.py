#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from htce_origin import HTCERuntime, RuntimeRequest

runtime = HTCERuntime()
print(runtime.wake().output)
for text in (
    "FACT Mary located_in office EVID event42",
    "QUERY Mary location EVID ask42",
    "FACT Mary located_in garden EVID event43",
    "QUERY Mary location EVID ask43",
    "QUERY John location EVID ask99",
):
    response = runtime.tick(RuntimeRequest(text))
    print(f"> {text}")
    print(response.output)
print(runtime.health())
