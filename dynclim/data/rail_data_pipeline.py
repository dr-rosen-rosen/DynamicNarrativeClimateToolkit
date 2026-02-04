"""
Streamlined Rail Safety Data Pipeline
=====================================

Single-file pipeline for loading, cleaning, and preparing FRA rail safety data
for safety climate feature extraction.

Data Sources (all CSV downloads from FRA):
- Event reports: Form 54/55/57 accident/incident reports with narratives
- Operational data: Monthly train_miles, passenger_miles, staff_hours by operator

Outputs:
- events.parquet: Cleaned event reports ready for NLP feature extraction
- operational.parquet: Monthly operational covariates by organization
- combined.parquet: Events joined with lagged operational data (optional)

Usage:
    from rail_data_pipeline import RailDataPipeline
    
    pipeline = RailDataPipeline(
        events_path="data/raw/rail/fra_events.csv",
        ops_path="data/raw/rail/fra_operational.csv",
        output_dir="data/processed/rail"
    )
    
    # Load and clean
    events_df = pipeline.load_events()
    ops_df = pipeline.load_operational()
    
    # Or run full pipeline
    pipeline.run()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, field
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RailConfig:
    """Configuration for Rail data pipeline."""
    
    # Column mappings: standardized name -> raw column name(s)
    # Events - Form 54 specific
    event_id_col: str = "INCDTNO"                 # FRA incident number (raw event ID)
    org_id_col: str = "Reporting Railroad Code"                  # Railroad/operator code  
    
    # Narrative columns to concatenate (Form 54)
    narrative_cols: List[str] = field(default_factory=lambda: [
        "NARR1", "NARR2", "NARR3", "NARR4", "NARR5", 
        "NARR6", "NARR7", "NARR8", "NARR9", "NARR10",
        "NARR11", "NARR12", "NARR13", "NARR14", "NARR15"
    ])
    
    # Date component columns (Form 54 uses separate YEAR, MONTH, DAY)
    year_col: str = "IYR"       # 2-digit year
    month_col: str = "IMO"      # Month (integer)
    day_col: str = "DAY"        # Day (2-digit)
    
    # Alternative column names (FRA formats vary across forms)
    event_id_alts: List[str] = field(default_factory=lambda: [
        "INCDTNO", "GXID", "AccidentNumber", "Accident.Number", "eid"
    ])
    org_id_alts: List[str] = field(default_factory=lambda: [
        "RAILROAD", "Railroad", "RR", "org_id", "operator", "Reporting Railroad Code"
    ])
    
    # Outcomes (for joining later)
    injury_col: str = "Total Persons Injured"
    fatality_col: str = "Total Persons Killed"  
    damage_col: str = "Total Damage Cost"
    
    # Operational data columns
    ops_date_col: str = "yearmonth"
    ops_org_col: str = "RAILROAD"
    train_miles_col: str = "train_miles"
    passenger_miles_col: str = "passenger_miles"
    staff_hours_col: str = "employee_hours"
    
    # Cleaning parameters
    min_narrative_words: int = 5
    min_narrative_chars: int = 20
    
    # Output
    output_format: str = "parquet"  # parquet, csv, or feather


# =============================================================================
# Core Pipeline
# =============================================================================

class RailDataPipeline:
    """
    Streamlined pipeline for FRA Rail safety data.
    
    Handles:
    - Loading event reports (Form 54/55/57 combined)
    - Loading operational data (train miles, passenger miles, staff hours)
    - Cleaning and standardizing column names
    - Filtering invalid/empty narratives
    - Joining events with operational covariates
    """
    
    def __init__(
        self,
        events_path: Optional[Union[str, Path]] = None,
        ops_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "data/processed/rail",
        config: Optional[RailConfig] = None
    ):
        """
        Initialize Rail pipeline.
        
        Args:
            events_path: Path to events CSV file
            ops_path: Path to operational data CSV file
            output_dir: Directory for output files
            config: Optional RailConfig for customization
        """
        self.events_path = Path(events_path) if events_path else None
        self.ops_path = Path(ops_path) if ops_path else None
        self.output_dir = Path(output_dir)
        self.config = config or RailConfig()
        
        # State
        self._events_df: Optional[pd.DataFrame] = None
        self._ops_df: Optional[pd.DataFrame] = None
        
    # -------------------------------------------------------------------------
    # Event Data
    # -------------------------------------------------------------------------
    
    def load_events(
        self,
        path: Optional[Union[str, Path]] = None,
        clean: bool = True
    ) -> pd.DataFrame:
        """
        Load and optionally clean event reports.
        
        Args:
            path: Override path (uses self.events_path if None)
            clean: Whether to apply cleaning steps
            
        Returns:
            DataFrame with event reports
        """
        path = Path(path) if path else self.events_path
        if path is None:
            raise ValueError("No events path specified")
        
        logger.info(f"Loading events from {path}")
        
        # Read CSV with flexible encoding
        try:
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        
        logger.info(f"Loaded {len(df):,} raw event records")
        
        # Standardize column names
        df = self._standardize_event_columns(df)
        
        if clean:
            df = self.clean_events(df)
        
        self._events_df = df
        return df
    
    def _standardize_event_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map varied column names to standardized names.
        
        Handles Form 54 specifics:
        - Generates UUID for each event (eid)
        - Preserves raw event ID (raw_event_id from INCDTNO)
        - Concatenates NARR1-NARR15 into narrative
        - Constructs event_date from YEAR, MONTH, DAY components
        """
        import uuid
        
        df = df.copy()
        
        # --- 1. Event ID: find raw ID and generate UUID ---
        raw_id_col = self._find_column(df, self.config.event_id_alts, self.config.event_id_col)
        if raw_id_col and raw_id_col in df.columns:
            # Keep raw event ID
            df['raw_event_id'] = df[raw_id_col].astype(str)
            logger.info(f"Found raw event ID column: '{raw_id_col}'")
        else:
            # Fallback: use index as raw ID
            df['raw_event_id'] = df.index.astype(str)
            logger.warning("No raw event ID column found, using index")
        
        # Generate UUID for each event
        df['eid'] = [str(uuid.uuid4()) for _ in range(len(df))]
        logger.info(f"Generated UUIDs for {len(df):,} events")
        
        # --- 2. Organization ID ---
        org_col = self._find_column(df, self.config.org_id_alts, self.config.org_id_col)
        if org_col and org_col in df.columns and org_col != 'org_id':
            df = df.rename(columns={org_col: 'org_id'})
            logger.debug(f"Mapped '{org_col}' -> 'org_id'")
        
        # --- 3. Narrative: concatenate NARR1-NARR15 ---
        narr_cols = [c for c in self.config.narrative_cols if c in df.columns]
        if narr_cols:
            logger.info(f"Concatenating {len(narr_cols)} narrative columns: {narr_cols[0]}...{narr_cols[-1]}")
            df['narrative'] = (
                df[narr_cols]
                .fillna('')
                .astype(str)
                .apply(lambda row: ' '.join(row).strip(), axis=1)
            )
            # Clean up multiple spaces
            df['narrative'] = df['narrative'].str.replace(r'\s+', ' ', regex=True).str.strip()
        else:
            # Fallback: look for existing narrative column
            narr_alts = ['Narrative', 'NARR', 'narrative', 'cmbd_narrative']
            found_narr = self._find_column(df, narr_alts, 'Narrative')
            if found_narr and found_narr in df.columns:
                df = df.rename(columns={found_narr: 'narrative'})
                logger.info(f"Using existing narrative column: '{found_narr}'")
            else:
                logger.warning("No narrative columns found!")
        
        # --- 4. Date: construct from YEAR, MONTH, DAY components ---
        year_col = self._find_column(df, [self.config.year_col, 'YEAR', 'IYR', 'YR'], self.config.year_col)
        month_col = self._find_column(df, [self.config.month_col, 'MONTH', 'IMO', 'MO'], self.config.month_col)
        day_col = self._find_column(df, [self.config.day_col, 'DAY'], self.config.day_col)
        
        if year_col and month_col and day_col:
            if all(c in df.columns for c in [year_col, month_col, day_col]):
                logger.info(f"Constructing date from: {year_col}, {month_col}, {day_col}")
                df['event_date'] = self._construct_date(
                    df[year_col], df[month_col], df[day_col]
                )
        else:
            # Fallback: look for existing date column
            date_alts = ['DATE', 'Date', 'event_date', 'date']
            found_date = self._find_column(df, date_alts, 'DATE')
            if found_date and found_date in df.columns:
                df['event_date'] = pd.to_datetime(df[found_date], errors='coerce')
                logger.info(f"Using existing date column: '{found_date}'")
        
        # --- 5. Outcome columns ---
        outcome_mappings = [
            ('total_injured', ['TOTINJ', 'TotalInjured', 'total_persons_injured']),
            ('total_killed', ['TOTKLD', 'TotalKilled', 'total_persons_killed']),
            ('total_damage', ['ACCDMG', 'TotalDamage', 'total_damage_cost']),
        ]
        for std_name, alts in outcome_mappings:
            found = self._find_column(df, alts, alts[0])
            if found and found in df.columns and found != std_name:
                df = df.rename(columns={found: std_name})
        
        return df
    
    def _construct_date(
        self, 
        year_series: pd.Series, 
        month_series: pd.Series, 
        day_series: pd.Series
    ) -> pd.Series:
        """
        Construct date from separate YEAR, MONTH, DAY columns.
        
        Handles 2-digit years: 00-29 -> 2000-2029, 30-99 -> 1930-1999
        """
        def convert_year(y):
            """Convert 2-digit year to 4-digit."""
            if pd.isna(y):
                return np.nan
            try:
                y = int(y)
                if y < 100:
                    # 2-digit year: 00-29 -> 2000s, 30-99 -> 1900s
                    return 2000 + y if y < 30 else 1900 + y
                return y  # Already 4-digit
            except (ValueError, TypeError):
                return np.nan
        
        years = year_series.apply(convert_year)
        
        # Build date strings and parse
        date_strs = (
            years.astype(str).str.replace('.0', '', regex=False) + '-' +
            month_series.astype(str).str.zfill(2) + '-' +
            day_series.astype(str).str.zfill(2)
        )
        
        return pd.to_datetime(date_strs, format='%Y-%m-%d', errors='coerce')
    
    def _find_column(
        self, 
        df: pd.DataFrame, 
        alternatives: List[str], 
        default: str
    ) -> Optional[str]:
        """Find first matching column from alternatives list."""
        # Check default first
        if default in df.columns:
            return default
        
        # Check alternatives (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}
        for alt in alternatives:
            if alt.lower() in cols_lower:
                return cols_lower[alt.lower()]
        
        return None
    
    def clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean event data.
        
        Steps:
        1. Remove rows with missing/empty narratives
        2. Clean narrative text (remove "NoneNone" artifacts, etc.)
        3. Filter short narratives
        4. Validate dates (already parsed in standardize step)
        5. Ensure valid UTF-8
        6. Drop duplicates on raw_event_id (keep UUID mapping)
        """
        initial_count = len(df)
        df = df.copy()
        
        # 1. Basic narrative filtering
        if 'narrative' in df.columns:
            # Remove null/empty
            df = df[df['narrative'].notna()]
            df = df[df['narrative'].astype(str).str.strip() != '']
            
            # Clean text artifacts (common in FRA data)
            df['narrative'] = (df['narrative']
                .astype(str)
                .str.replace('NoneNone', '', regex=False)
                .str.replace(r'None$', '', regex=True)
                .str.replace(r'^\s*None\s*$', '', regex=True)
                .str.strip()
            )
            
            # Filter empty after cleaning
            df = df[df['narrative'] != '']
            df = df[df['narrative'].str.lower() != 'none']
            
            # 2. Filter short narratives
            df['_word_count'] = df['narrative'].str.split().str.len()
            df = df[df['_word_count'] >= self.config.min_narrative_words]
            df = df[df['narrative'].str.len() >= self.config.min_narrative_chars]
            df = df.drop(columns=['_word_count'])
            
            # 3. UTF-8 validation
            df = df[df['narrative'].apply(self._is_valid_utf8)]
        
        # 4. Date validation (already parsed, just check for valid dates)
        if 'event_date' in df.columns:
            n_invalid_dates = df['event_date'].isna().sum()
            if n_invalid_dates > 0:
                logger.warning(f"{n_invalid_dates:,} events have invalid/missing dates")
            # Optionally filter bad dates:
            # df = df[df['event_date'].notna()]
        
        # 5. Deduplicate on raw_event_id (preserves UUID mapping)
        if 'raw_event_id' in df.columns:
            n_before = len(df)
            df = df.drop_duplicates(subset=['raw_event_id'], keep='first')
            n_dupes = n_before - len(df)
            if n_dupes > 0:
                logger.info(f"Removed {n_dupes:,} duplicate events (by raw_event_id)")
        elif 'eid' in df.columns:
            df = df.drop_duplicates(subset=['eid'], keep='first')
        
        final_count = len(df)
        logger.info(f"Cleaning: {initial_count:,} -> {final_count:,} events "
                   f"({initial_count - final_count:,} removed)")
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def _is_valid_utf8(text: str) -> bool:
        """Check if text is valid UTF-8."""
        if pd.isna(text):
            return False
        try:
            str(text).encode('utf-8')
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False
    
    # -------------------------------------------------------------------------
    # Operational Data
    # -------------------------------------------------------------------------
    
    def load_operational(
        self,
        path: Optional[Union[str, Path]] = None,
        clean: bool = True
    ) -> pd.DataFrame:
        """
        Load operational data (train miles, passenger miles, staff hours).
        
        Args:
            path: Override path (uses self.ops_path if None)
            clean: Whether to apply cleaning steps
            
        Returns:
            DataFrame with operational data by org/month
        """
        path = Path(path) if path else self.ops_path
        if path is None:
            raise ValueError("No operational data path specified")
        
        logger.info(f"Loading operational data from {path}")
        
        try:
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        
        logger.info(f"Loaded {len(df):,} operational records")
        
        # Standardize columns
        df = self._standardize_ops_columns(df)
        
        if clean:
            df = self.clean_operational(df)
        
        self._ops_df = df
        return df
    
    def _standardize_ops_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map operational data columns to standard names."""
        df = df.copy()
        
        # Date column
        date_alts = ['yearmonth', 'YEARMONTH', 'date', 'DATE', 'year_month', 'ym']
        found = self._find_column(df, date_alts, self.config.ops_date_col)
        if found and found != 'yearmonth':
            df = df.rename(columns={found: 'yearmonth'})
        
        # Org column  
        org_alts = ['RAILROAD', 'Railroad', 'RR', 'org_id', 'operator', 'railroad']
        found = self._find_column(df, org_alts, self.config.ops_org_col)
        if found and found != 'org_id':
            df = df.rename(columns={found: 'org_id'})
        
        # Operational variables
        ops_mappings = [
            ('train_miles', ['train_miles', 'TRAIN_MILES', 'trainmiles', 'TM']),
            ('passenger_miles', ['passenger_miles', 'PASSENGER_MILES', 'passmiles', 'PM']),
            ('staff_hours', ['employee_hours', 'staff_hours', 'EMPLOYEE_HOURS', 'EMP_HOURS', 'hours']),
        ]
        for std_name, alts in ops_mappings:
            found = self._find_column(df, alts, alts[0])
            if found and found in df.columns and found != std_name:
                df = df.rename(columns={found: std_name})
        
        return df
    
    def clean_operational(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean operational data.
        
        Steps:
        1. Parse date to period
        2. Ensure numeric types for operational vars
        3. Handle missing values
        4. Aggregate duplicates if any
        """
        df = df.copy()
        
        # Parse yearmonth
        if 'yearmonth' in df.columns:
            # Handle various date formats
            df['yearmonth'] = pd.to_datetime(df['yearmonth'], errors='coerce')
            # Floor to month start
            df['yearmonth'] = df['yearmonth'].dt.to_period('M').dt.to_timestamp()
        
        # Ensure numeric
        ops_cols = ['train_miles', 'passenger_miles', 'staff_hours']
        for col in ops_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Aggregate by org/month (sum) if duplicates
        group_cols = ['org_id', 'yearmonth']
        if all(c in df.columns for c in group_cols):
            agg_cols = {c: 'sum' for c in ops_cols if c in df.columns}
            if agg_cols:
                df = df.groupby(group_cols, as_index=False).agg(agg_cols)
        
        logger.info(f"Operational data: {len(df):,} org-month records")
        return df
    
    # -------------------------------------------------------------------------
    # Combined / Join
    # -------------------------------------------------------------------------
    
    def join_events_with_ops(
        self,
        events_df: Optional[pd.DataFrame] = None,
        ops_df: Optional[pd.DataFrame] = None,
        lag_months: int = 1
    ) -> pd.DataFrame:
        """
        Join events with lagged operational data.
        
        Args:
            events_df: Events DataFrame (uses cached if None)
            ops_df: Operational DataFrame (uses cached if None)
            lag_months: How many months to lag operational data
            
        Returns:
            Events with operational covariates joined
        """
        events_df = events_df if events_df is not None else self._events_df
        ops_df = ops_df if ops_df is not None else self._ops_df
        
        if events_df is None or ops_df is None:
            raise ValueError("Both events and operational data required for join")
        
        df = events_df.copy()
        
        # Create yearmonth from event date
        if 'event_date' in df.columns:
            df['_event_ym'] = pd.to_datetime(df['event_date']).dt.to_period('M').dt.to_timestamp()
            # Lag: operational data from (event_month - lag_months)
            df['_join_ym'] = df['_event_ym'] - pd.DateOffset(months=lag_months)
        else:
            raise ValueError("event_date column required for join")
        
        # Prepare ops for join
        ops = ops_df.copy()
        ops = ops.rename(columns={'yearmonth': '_join_ym'})
        
        # Join
        df = df.merge(
            ops,
            on=['org_id', '_join_ym'],
            how='left'
        )
        
        # Clean up temp columns
        df = df.drop(columns=['_event_ym', '_join_ym'], errors='ignore')
        
        n_matched = df[['train_miles', 'passenger_miles', 'staff_hours']].notna().any(axis=1).sum()
        logger.info(f"Joined operational data: {n_matched:,}/{len(df):,} events matched")
        
        return df
    
    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    
    def save(
        self,
        df: pd.DataFrame,
        name: str,
        format: Optional[str] = None
    ) -> Path:
        """
        Save DataFrame to output directory.
        
        Args:
            df: DataFrame to save
            name: Base filename (without extension)
            format: Override output format
            
        Returns:
            Path to saved file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fmt = format or self.config.output_format
        
        if fmt == 'parquet':
            path = self.output_dir / f"{name}.parquet"
            df.to_parquet(path, index=False, compression='zstd')
        elif fmt == 'feather':
            path = self.output_dir / f"{name}.feather"
            df.to_feather(path)
        elif fmt == 'csv':
            path = self.output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unknown format: {fmt}")
        
        logger.info(f"Saved {len(df):,} records to {path}")
        return path
    
    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------
    
    def run(
        self,
        save_outputs: bool = True,
        join_operational: bool = True,
        lag_months: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Run full pipeline.
        
        Args:
            save_outputs: Whether to save to disk
            join_operational: Whether to join ops with events
            lag_months: Lag for operational data join
            
        Returns:
            Dict with 'events', 'operational', and optionally 'combined' DataFrames
        """
        logger.info("=" * 60)
        logger.info("Running Rail Data Pipeline")
        logger.info("=" * 60)
        
        results = {}
        
        # Load events
        if self.events_path:
            events_df = self.load_events()
            results['events'] = events_df
            if save_outputs:
                self.save(events_df, 'events')
        
        # Load operational
        if self.ops_path:
            ops_df = self.load_operational()
            results['operational'] = ops_df
            if save_outputs:
                self.save(ops_df, 'operational')
        
        # Join if requested and both available
        if join_operational and self._events_df is not None and self._ops_df is not None:
            combined_df = self.join_events_with_ops(lag_months=lag_months)
            results['combined'] = combined_df
            if save_outputs:
                self.save(combined_df, 'events_with_ops')
        
        logger.info("=" * 60)
        logger.info("Pipeline complete")
        logger.info("=" * 60)
        
        return results
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def get_report_data(self) -> List[Tuple[str, str]]:
        """
        Get data in format expected by feature extraction pipeline.
        
        Returns:
            List of (report_id, narrative_text) tuples
        """
        if self._events_df is None:
            raise ValueError("Load events first with load_events()")
        
        df = self._events_df
        if 'eid' not in df.columns or 'narrative' not in df.columns:
            raise ValueError("Required columns 'eid' and 'narrative' not found")
        
        return list(zip(
            df['eid'].astype(str),
            df['narrative'].astype(str)
        ))
    
    def summary(self) -> Dict:
        """Get summary statistics for loaded data."""
        stats = {}
        
        if self._events_df is not None:
            df = self._events_df
            stats['events'] = {
                'n_records': len(df),
                'n_orgs': df['org_id'].nunique() if 'org_id' in df.columns else None,
                'date_range': (
                    str(df['event_date'].min()),
                    str(df['event_date'].max())
                ) if 'event_date' in df.columns else None,
                'narrative_words_mean': df['narrative'].str.split().str.len().mean() if 'narrative' in df.columns else None,
            }
        
        if self._ops_df is not None:
            df = self._ops_df
            stats['operational'] = {
                'n_records': len(df),
                'n_orgs': df['org_id'].nunique() if 'org_id' in df.columns else None,
                'date_range': (
                    str(df['yearmonth'].min()),
                    str(df['yearmonth'].max())
                ) if 'yearmonth' in df.columns else None,
            }
        
        return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process FRA Rail safety data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process events only
    python rail_data_pipeline.py --events data/raw/fra_events.csv
    
    # Process both events and operational data
    python rail_data_pipeline.py \\
        --events data/raw/fra_events.csv \\
        --ops data/raw/fra_operational.csv \\
        --output data/processed/rail
    
    # Just clean and validate, don't save
    python rail_data_pipeline.py --events data.csv --no-save --summary
        """
    )
    
    parser.add_argument('--events', '-e', help='Path to events CSV')
    parser.add_argument('--ops', '-o', help='Path to operational data CSV')
    parser.add_argument('--output', '-d', default='data/processed/rail',
                       help='Output directory (default: data/processed/rail)')
    parser.add_argument('--format', '-f', choices=['parquet', 'csv', 'feather'],
                       default='parquet', help='Output format')
    parser.add_argument('--no-save', action='store_true', help='Do not save outputs')
    parser.add_argument('--no-join', action='store_true', 
                       help='Do not join operational with events')
    parser.add_argument('--lag', type=int, default=1,
                       help='Months to lag operational data (default: 1)')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Print summary statistics')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.events and not args.ops:
        parser.error("At least one of --events or --ops required")
    
    # Configure and run
    config = RailConfig(output_format=args.format)
    
    pipeline = RailDataPipeline(
        events_path=args.events,
        ops_path=args.ops,
        output_dir=args.output,
        config=config
    )
    
    results = pipeline.run(
        save_outputs=not args.no_save,
        join_operational=not args.no_join,
        lag_months=args.lag
    )
    
    if args.summary:
        import json
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(json.dumps(pipeline.summary(), indent=2, default=str))


if __name__ == "__main__":
    main()
