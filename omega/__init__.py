"""omega: a small vision model, served and instrumented.

Deliberately small and honest. This is not an ImageNet entry: it is a tiny CNN
trained on synthetic shapes, wrapped in the parts that real serving needs and
that most demos skip — input validation, a readiness gate, metrics, structured
logs, and a benchmark you can re-run.
"""

__version__ = "0.1.0"
SERVICE = "omega-vision"
