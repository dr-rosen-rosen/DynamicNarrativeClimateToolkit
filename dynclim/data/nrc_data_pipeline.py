"""
Streamlined NRC (Nuclear Regulatory Commission) Data Pipeline
=============================================================

Single-file pipeline for loading, scraping, cleaning, and preparing NRC 
Licensee Event Reports (LER) for safety climate feature extraction.

Data Sources:
- LER Search (lersearch.inl.gov): Requires Selenium for JS-rendered content
- Pre-scraped CSV: For re-processing without re-scraping

Workflow:
1. Get event numbers from NRC search (manual export to Excel)
2. Scrape event details from LER Search (or load from CSV)
3. Clean and standardize
4. Output for feature extraction

Usage:
    from nrc_data_pipeline import NRCDataPipeline
    
    # From pre-scraped data
    pipeline = NRCDataPipeline(
        events_path="data/raw/nrc/nrc_events.csv",
        output_dir="data/processed/nrc"
    )
    events_df = pipeline.load_events()
    
    # With scraping
    pipeline = NRCDataPipeline(output_dir="data/processed/nrc")
    events_df = pipeline.scrape_events("ENSearchResults.xlsx")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Set
from dataclasses import dataclass, field
import logging
import pickle
import time
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class NRCConfig:
    """Configuration for NRC data pipeline."""
    
    # LER Search URL pattern
    ler_base_url: str = "https://lersearch.inl.gov/ENView.aspx?DOC::"
    
    # Field IDs on LER page (Selenium element IDs)
    field_ids: Dict[str, str] = field(default_factory=lambda: {
        'event_num': "ContentPlaceHolderMainPageContent_Label1EventNumber",
        'facility': "ContentPlaceHolderMainPageContent_Label1Facility",
        'emerg_class': "ContentPlaceHolderMainPageContent_Label1EmergencyClass",
        '10_cfr_sec': "ContentPlaceHolderMainPageContent_Label110CFRSection",
        'last_update_date': "ContentPlaceHolderMainPageContent_Label1LastUpdateDate",
        'event_date': "ContentPlaceHolderMainPageContent_Label1EventDate",
        'event_time': "ContentPlaceHolderMainPageContent_Label1EventTime",
        'notification_date': "ContentPlaceHolderMainPageContent_Label1NotificationDate",
        'notification_time': "ContentPlaceHolderMainPageContent_Label1NotificationTime",
        'region': "ContentPlaceHolderMainPageContent_Label1Region",
        'state': "ContentPlaceHolderMainPageContent_Label1State",
        'rx_type': "ContentPlaceHolderMainPageContent_Label1RXType",
        'unit': "ContentPlaceHolderMainPageContent_LabelUnit1",
        'scram_code': "ContentPlaceHolderMainPageContent_LabelSCRAMCode1",
        'rx_crit': "ContentPlaceHolderMainPageContent_LabelRXCrit1",
        'initial_pwr': "ContentPlaceHolderMainPageContent_LabelInitialPower1",
        'initial_rx_mode': "ContentPlaceHolderMainPageContent_LabelInitialRXMode1",
        'current_pwr': "ContentPlaceHolderMainPageContent_LabelCurrentPower1",
        'current_rx_mode': "ContentPlaceHolderMainPageContent_LabelCurrentRXMode1",
        'event_text': "ContentPlaceHolderMainPageContent_LabelEventText",
    })
    
    # Scraping settings
    delay_between_requests: float = 1.0
    save_interval: int = 50
    max_retries: int = 2
    headless: bool = True
    
    # Cleaning settings
    min_narrative_words: int = 5
    min_narrative_chars: int = 20
    
    # Known data corrections
    date_corrections: Dict[str, str] = field(default_factory=lambda: {
        '11/20/2103': '11/20/2013',  # Known typo in NRC data
    })
    
    # Facility name standardization
    facility_mapping: Dict[str, str] = field(default_factory=lambda: {
        'columbia generating statiregion:': 'columbia generating station',
        'davis-besse': 'davis besse',
        'washington nuclear (wnp-2region:': 'washington nuclear',
        'vogtle 1/2': 'vogtle',
        'vogtle 3/4': 'vogtle',
        'fort calhoun': 'ft calhoun',
        'summer construction': 'summer',
    })
    
    # Output
    output_format: str = "parquet"


# =============================================================================
# Core Pipeline
# =============================================================================

class NRCDataPipeline:
    """
    Streamlined pipeline for NRC Licensee Event Reports.
    
    Handles:
    - Scraping from LER Search (Selenium-based)
    - Loading from pre-scraped CSV
    - Cleaning and standardization
    - Facility name normalization
    - Date parsing and correction
    """
    
    def __init__(
        self,
        events_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "data/processed/nrc",
        config: Optional[NRCConfig] = None
    ):
        """
        Initialize NRC pipeline.
        
        Args:
            events_path: Path to pre-scraped events CSV (if available)
            output_dir: Directory for output files
            config: Optional NRCConfig for customization
        """
        self.events_path = Path(events_path) if events_path else None
        self.output_dir = Path(output_dir)
        self.config = config or NRCConfig()
        
        # State
        self._events_df: Optional[pd.DataFrame] = None
        self._checkpoint: Dict = {'processed': set(), 'failed': set(), 'data': []}
        
    # -------------------------------------------------------------------------
    # Loading (from CSV)
    # -------------------------------------------------------------------------
    
    def load_events(
        self,
        path: Optional[Union[str, Path]] = None,
        clean: bool = True
    ) -> pd.DataFrame:
        """
        Load events from CSV file.
        
        Args:
            path: Override path (uses self.events_path if None)
            clean: Whether to apply cleaning steps
            
        Returns:
            DataFrame with event reports
        """
        path = Path(path) if path else self.events_path
        if path is None:
            raise ValueError("No events path specified")
        
        logger.info(f"Loading NRC events from {path}")
        
        try:
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        
        # Drop unnamed index columns
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        
        logger.info(f"Loaded {len(df):,} raw event records")
        
        if clean:
            df = self.clean_events(df)
        
        self._events_df = df
        return df
    
    # -------------------------------------------------------------------------
    # Scraping (Selenium)
    # -------------------------------------------------------------------------
    
    def scrape_events(
        self,
        event_nums_file: Union[str, Path],
        resume: bool = True,
        save_intermediate: bool = True,
        retry_failed: bool = False
    ) -> pd.DataFrame:
        """
        Scrape event details from LER Search.

        Args:
            event_nums_file: Excel file with event numbers (from NRC search export)
            resume: Whether to resume from checkpoint
            save_intermediate: Whether to save checkpoint during scraping
            retry_failed: Re-attempt events in the checkpoint's failed set
                (otherwise resume skips them forever)

        Returns:
            DataFrame with scraped events
        """
        # Import Selenium only when needed
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "Scraping requires selenium and beautifulsoup4. "
                "Install with: pip install selenium beautifulsoup4"
            )
        
        # Load event numbers
        event_nums = self._load_event_numbers(event_nums_file)
        logger.info(f"Found {len(event_nums):,} event numbers to scrape")
        
        # Load checkpoint if resuming
        checkpoint_path = self.output_dir / "nrc_scraping_checkpoint.pkl"
        if resume and checkpoint_path.exists():
            self._checkpoint = self._load_checkpoint(checkpoint_path)
            logger.info(f"Resuming: {len(self._checkpoint['processed']):,} already done, "
                       f"{len(self._checkpoint['failed']):,} failed")
        
        # Filter to unprocessed
        if retry_failed and self._checkpoint['failed']:
            logger.info(f"Retrying {len(self._checkpoint['failed']):,} "
                        f"previously failed events")
            self._checkpoint['failed'] = set()
        to_scrape = [n for n in event_nums
                     if n not in self._checkpoint['processed']
                     and n not in self._checkpoint['failed']]
        logger.info(f"Scraping {len(to_scrape):,} remaining events")
        
        if not to_scrape:
            logger.info("All events already processed")
            return self._checkpoint_to_df()
        
        # Initialize browser
        browser = self._create_browser()
        
        try:
            for i, event_num in enumerate(to_scrape):
                try:
                    # Scrape single event
                    event_data = self._scrape_single_event(browser, event_num, BeautifulSoup)
                    
                    if event_data:
                        self._checkpoint['data'].append(event_data)
                        self._checkpoint['processed'].add(event_num)
                        logger.debug(f"[{i+1}/{len(to_scrape)}] {event_num} - OK")
                    else:
                        self._checkpoint['failed'].add(event_num)
                        logger.warning(f"[{i+1}/{len(to_scrape)}] {event_num} - Failed (no data)")
                    
                except Exception as e:
                    self._checkpoint['failed'].add(event_num)
                    logger.warning(f"[{i+1}/{len(to_scrape)}] {event_num} - Error: {e}")
                
                # Save checkpoint periodically
                if save_intermediate and (i + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(checkpoint_path)
                    logger.info(f"Checkpoint saved: {len(self._checkpoint['processed']):,} processed")
                
                # Rate limiting
                time.sleep(self.config.delay_between_requests)
                
        finally:
            browser.quit()
            # Final save
            self._save_checkpoint(checkpoint_path)
        
        # Convert to DataFrame
        df = self._checkpoint_to_df()
        
        # Save results
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.output_dir / "nrc_events_raw.csv"
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved {len(df):,} raw events to {raw_path}")
        
        # Save failed events list
        if self._checkpoint['failed']:
            failed_df = pd.DataFrame({'event_num': list(self._checkpoint['failed'])})
            failed_df.to_csv(self.output_dir / "failed_events.csv", index=False)
        
        # Clean and return
        df = self.clean_events(df)
        self._events_df = df
        return df
    
    def _load_event_numbers(self, file_path: Union[str, Path]) -> List[str]:
        """Load event numbers from Excel file."""
        file_path = Path(file_path)
        
        if file_path.suffix in ['.xlsx', '.xls']:
            # NRC search exports typically have headers in first few rows
            df = pd.read_excel(file_path, skiprows=4)
            # Find event number column
            for col in ['Event Number', 'event_num', 'EventNumber']:
                if col in df.columns:
                    return df[col].dropna().astype(str).tolist()
            # Try first column
            return df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            # CSV
            df = pd.read_csv(file_path)
            for col in ['Event Number', 'event_num', 'EventNumber']:
                if col in df.columns:
                    return df[col].dropna().astype(str).tolist()
            return df.iloc[:, 0].dropna().astype(str).tolist()
    
    def _create_browser(self):
        """Create Selenium browser instance."""
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options as FirefoxOptions
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        
        # Try Firefox first
        try:
            options = FirefoxOptions()
            if self.config.headless:
                options.add_argument('--headless')
            browser = webdriver.Firefox(options=options)
            logger.info("Using Firefox browser")
            return browser
        except Exception:
            pass
        
        # Fall back to Chrome
        try:
            options = ChromeOptions()
            if self.config.headless:
                options.add_argument('--headless')
            browser = webdriver.Chrome(options=options)
            logger.info("Using Chrome browser")
            return browser
        except Exception:
            raise RuntimeError(
                "Could not initialize browser. Install Firefox/geckodriver or Chrome/chromedriver."
            )
    
    def _scrape_single_event(self, browser, event_num: str, BeautifulSoup) -> Optional[Dict]:
        """Scrape a single event from LER Search."""
        url = f"{self.config.ler_base_url}{event_num}"
        
        browser.get(url)
        time.sleep(0.5)  # Wait for JS to render
        
        html = browser.page_source
        soup = BeautifulSoup(html, features="html.parser")
        
        event_data = {}
        for field_name, element_id in self.config.field_ids.items():
            element = soup.find(id=element_id)
            if element:
                # get_text with a separator: bare element.text concatenates
                # block elements and <br> with NO separator, fusing words
                # across boundaries ("NOZZLESTHE", title glued to body).
                # Collapse whitespace runs so line-wraps don't survive either.
                text = element.get_text(separator=" ")
                event_data[field_name] = " ".join(text.split())
            else:
                event_data[field_name] = None
        
        # Validate we got meaningful data
        if not event_data.get('event_num') or not event_data.get('event_text'):
            return None
        
        return event_data
    
    def _load_checkpoint(self, path: Path) -> Dict:
        """Load checkpoint from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _save_checkpoint(self, path: Path):
        """Save checkpoint to pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._checkpoint, f)
    
    def _checkpoint_to_df(self) -> pd.DataFrame:
        """Convert checkpoint data to DataFrame."""
        if self._checkpoint['data']:
            return pd.DataFrame(self._checkpoint['data'])
        return pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # Cleaning
    # -------------------------------------------------------------------------
    
    def clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NRC event data.
        
        Steps:
        1. Fix known date errors
        2. Parse dates
        3. Lowercase and clean event text
        4. Standardize facility names
        5. Filter invalid narratives
        6. Drop duplicates
        """
        initial_count = len(df)
        df = df.copy()
        
        # 1. Fix known date errors
        if 'event_date' in df.columns:
            for wrong, correct in self.config.date_corrections.items():
                df['event_date'] = df['event_date'].replace(wrong, correct)
        
        # 2. Parse dates
        date_cols = ['event_date', 'notification_date', 'last_update_date']
        for col in date_cols:
            if col in df.columns:
                # Try multiple formats
                for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y']:
                    try:
                        df[f'{col}_parsed'] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                        if df[f'{col}_parsed'].notna().sum() > 0:
                            df[col] = df[f'{col}_parsed']
                            df = df.drop(columns=[f'{col}_parsed'])
                            break
                    except:
                        continue
        
        # 3. Clean event text — case is PRESERVED (VADER and cased
        # transformer models are case-sensitive; spaCy syntactic features
        # need true casing; see project memory nrc-text-quality)
        if 'event_text' in df.columns:
            df['event_text'] = df['event_text'].astype(str).str.strip()
            
            # Filter empty/invalid (case-insensitive now that case is kept)
            df = df[df['event_text'].notna()]
            df = df[df['event_text'] != '']
            df = df[~df['event_text'].str.lower().isin(['none', 'nan'])]
            
            # Filter short
            df['_word_count'] = df['event_text'].str.split().str.len()
            df = df[df['_word_count'] >= self.config.min_narrative_words]
            df = df[df['event_text'].str.len() >= self.config.min_narrative_chars]
            df = df.drop(columns=['_word_count'])
        
        # 4. Standardize facility names
        if 'facility' in df.columns:
            df['facility'] = df['facility'].str.lower().str.strip()
            df['facility'] = df['facility'].replace(self.config.facility_mapping)
        
        # 5. Drop duplicates
        if 'event_num' in df.columns:
            df = df.drop_duplicates(subset=['event_num'], keep='first')
        
        final_count = len(df)
        logger.info(f"Cleaning: {initial_count:,} -> {final_count:,} events "
                   f"({initial_count - final_count:,} removed)")
        
        return df.reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    
    def save(
        self,
        df: pd.DataFrame,
        name: str,
        format: Optional[str] = None
    ) -> Path:
        """Save DataFrame to output directory."""
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
        event_nums_file: Optional[Union[str, Path]] = None,
        save_outputs: bool = True
    ) -> pd.DataFrame:
        """
        Run full pipeline.
        
        Args:
            event_nums_file: Excel file for scraping (if provided, scrapes; else loads)
            save_outputs: Whether to save to disk
            
        Returns:
            Cleaned events DataFrame
        """
        logger.info("=" * 60)
        logger.info("Running NRC Data Pipeline")
        logger.info("=" * 60)
        
        if event_nums_file:
            # Scrape mode
            events_df = self.scrape_events(event_nums_file)
        elif self.events_path:
            # Load mode
            events_df = self.load_events()
        else:
            raise ValueError("Either event_nums_file or events_path required")
        
        if save_outputs:
            self.save(events_df, 'events')
        
        logger.info("=" * 60)
        logger.info("Pipeline complete")
        logger.info("=" * 60)
        
        return events_df
    
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
            raise ValueError("Load events first")
        
        df = self._events_df
        id_col = 'event_num' if 'event_num' in df.columns else df.columns[0]
        text_col = 'event_text' if 'event_text' in df.columns else 'narrative'
        
        if text_col not in df.columns:
            raise ValueError(f"Required column '{text_col}' not found")
        
        return list(zip(
            df[id_col].astype(str),
            df[text_col].astype(str)
        ))
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        if self._events_df is None:
            return {}
        
        df = self._events_df
        return {
            'n_records': len(df),
            'n_facilities': df['facility'].nunique() if 'facility' in df.columns else None,
            'date_range': (
                str(df['event_date'].min()),
                str(df['event_date'].max())
            ) if 'event_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['event_date']) else None,
            'narrative_words_mean': df['event_text'].str.split().str.len().mean() if 'event_text' in df.columns else None,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process NRC nuclear event data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load from existing CSV
    python nrc_data_pipeline.py --events data/raw/nrc/nrc_events.csv
    
    # Scrape from LER Search
    python nrc_data_pipeline.py --scrape ENSearchResults.xlsx --headless
    
    # Resume interrupted scraping
    python nrc_data_pipeline.py --scrape ENSearchResults.xlsx --resume
        """
    )
    
    parser.add_argument('--events', '-e', help='Path to events CSV')
    parser.add_argument('--scrape', '-s', help='Excel file with event numbers to scrape')
    parser.add_argument('--output', '-d', default='data/processed/nrc',
                       help='Output directory')
    parser.add_argument('--format', '-f', choices=['parquet', 'csv', 'feather'],
                       default='parquet', help='Output format')
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests (seconds)')
    parser.add_argument('--no-save', action='store_true', help='Do not save outputs')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.events and not args.scrape:
        parser.error("Either --events or --scrape required")
    
    # Configure
    config = NRCConfig(
        output_format=args.format,
        headless=args.headless,
        delay_between_requests=args.delay
    )
    
    pipeline = NRCDataPipeline(
        events_path=args.events,
        output_dir=args.output,
        config=config
    )
    
    if args.scrape:
        events_df = pipeline.scrape_events(args.scrape, resume=args.resume)
    else:
        events_df = pipeline.load_events()
    
    if not args.no_save:
        pipeline.save(events_df, 'events')
    
    if args.summary:
        import json
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(json.dumps(pipeline.summary(), indent=2, default=str))


if __name__ == "__main__":
    main()
