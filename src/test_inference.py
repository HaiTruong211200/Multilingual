"""Backward-compatible entrypoint; inference now lives in :mod:`eval.inference`."""

from eval.inference import main


if __name__ == "__main__":
    main()
