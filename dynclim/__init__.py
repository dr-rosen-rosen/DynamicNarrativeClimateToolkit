# dynclim/__init__.py
"""
Dynamic Climate Safety Analysis Toolkit

A comprehensive toolkit for analyzing safety climate in dynamic systems
using early warning signals and machine learning approaches.
"""

__version__ = "0.2.0"

# Import the main classes users will need
from dynclim.features.safety import Processor
from dynclim.experiments import ExperimentRunner
from dynclim.data import get_reports_from_db, create_database_engine

__all__ = [
    'Processor',
    'ExperimentRunner',
    'get_reports_from_db',
    'create_database_engine'
]