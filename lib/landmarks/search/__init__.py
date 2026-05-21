"""Landmark candidate search subsystem.

Lives outside :mod:`lib.landmarks.eval` because candidate enumeration,
geometry evaluation, promotion gates, and report writers form a search/
promotion subsystem — they're not pure metric calculations.
"""
