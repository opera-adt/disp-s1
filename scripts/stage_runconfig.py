#!/usr/bin/env python3
"""
OPERA DISP-S1 Data Staging Script

This script stages all necessary data for OPERA DISP-S1 processing by 
reading exisitn Runconfig.yaml, including:
- CSLC and Compressed CSLC files
- Static layers
- DEM and water masks
- Ionosphere files
- Configuration files

It validates the input runconfig, downloads all required data, and generates
an updated runconfig with local file paths.
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import s3fs
import yaml
from botocore.session import Session
from osgeo import gdal

# Import OPERA utilities
import opera_utils
from opera_utils.download import download_cslc_static_layers, download_cslcs

# Try to import from disp_s1 package, fall back to local modules
try:
    from disp_s1._dem import S3_DEM_BUCKET, S3_LONLAT_VRT_KEY, stage_dem
    from disp_s1.ionosphere import (
        DEFAULT_DOWNLOAD_ENDPOINT,
        DownloadConfig,
        IonosphereType,
        download_ionosphere_files,
    )
    DISP_S1_INSTALLED = True
except ImportError:
    # If disp_s1 is not installed, we'll need to handle this
    DISP_S1_INSTALLED = False
    # Set default values
    S3_DEM_BUCKET = "opera-dem"
    S3_LONLAT_VRT_KEY = "v1.0/EPSG4326/EPSG4326.vrt"
    DEFAULT_DOWNLOAD_ENDPOINT = "https://cddis.nasa.gov/archive/gnss/products/ionex/"

# Import local water mask module
try:
    from water_mask import create_water_mask
except ImportError:
    # Try to import from disp_s1 if local module not available
    try:
        from disp_s1._water import create_water_mask
    except ImportError:
        create_water_mask = None

# Enable GDAL exceptions
gdal.UseExceptions()

# Constants
DB_FILENAME = "opera-disp-s1-consistent-burst-ids-2025-09-16-2016-07-01_to_2024-12-31.json"
JSON_URL = f"https://github.com/opera-adt/burst_db/releases/download/v0.13.0/{DB_FILENAME}"
DISP_CONFIG_URL = "https://raw.githubusercontent.com/opera-adt/disp-s1/main/configs/"
BURST_DB_URL = "https://github.com/opera-adt/burst_db/releases/download/"


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Suppress verbose logging from third-party libraries
    logging.getLogger("asf_search").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("s3fs").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


class DataStager:
    """Handles staging of OPERA DISP-S1 data."""

    def __init__(self, runconfig_path: Path, output_dir: Path, aws_profile: str = "saml-pub"):
        "Initialize the staging object with configuration paths and AWS profile."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.runconfig_path = runconfig_path
        self.output_dir = output_dir
        self.aws_profile = aws_profile
        
        # Load runconfig
        self.logger.info(f"Loading runconfig from {runconfig_path}")
        self.runconfig = self._load_runconfig()
        self.config_type = "SAS"  # Default to SAS config
        
        # Extract key parameters
        self.frame_id = self.runconfig["RunConfig"]["Groups"][self.config_type][
            "input_file_group"
        ]["frame_id"]
        self.input_list = self.runconfig["RunConfig"]["Groups"][self.config_type][
            "input_file_group"
        ]["cslc_file_list"]
        
        # Setup directory structure
        self._setup_directories()

    def _load_runconfig(self) -> dict:
        """Load and validate the runconfig YAML file."""
        with open(self.runconfig_path, "r") as f:
            try:
                data = yaml.safe_load(f)
                return data
            except yaml.YAMLError as exc:
                self.logger.error(f"Error loading runconfig: {exc}")
                raise

    def _setup_directories(self):
        """Create directory structure for staged data."""
        self.frame_dir = self.output_dir / f"F{self.frame_id}"
        self.input_dir = self.frame_dir / "input_groups"
        
        self.config_dir = self.input_dir / "configs"
        self.cslc_dir = self.input_dir / "input_cslc"
        self.ccslc_dir = self.input_dir / "input_ccslc"
        self.static_dir = self.input_dir / "input_static"
        self.iono_dir = self.static_dir / "ionosphere"
        
        # Create all directories
        for directory in [
            self.config_dir,
            self.cslc_dir,
            self.ccslc_dir,
            self.static_dir,
            self.iono_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")

    @staticmethod
    def _get_cslc_df(input_list: list) -> pd.DataFrame:
        """Parse CSLC filenames into a DataFrame."""
        cslc = [ix for ix in input_list if "_CSLC-S1_" in Path(ix).stem]
        
        columns = [
            "project",
            "level",
            "product",
            "burst_id",
            "date",
            "production_date",
            "sensor",
            "polarization",
            "version",
        ]
        
        cslc_df = pd.DataFrame(
            [Path(ix).stem.split("_") for ix in cslc], columns=columns
        )
        cslc_df["file"] = [Path(ix).stem for ix in cslc]
        cslc_df["yyyymmdd"] = pd.to_datetime(
            cslc_df["date"], format="%Y%m%dT%H%M%SZ", errors="raise"
        )
        cslc_df["yyyymmdd"] = cslc_df.yyyymmdd.dt.date
        
        return cslc_df

    @staticmethod
    def _get_ccslc_df(input_list: list) -> pd.DataFrame:
        """Parse Compressed CSLC filenames into a DataFrame."""
        compressed_cslc = [
            ix for ix in input_list if "_COMPRESSED-CSLC-S1_" in Path(ix).stem
        ]
        
        columns = [
            "project",
            "level",
            "product",
            "frame",
            "burst_id",
            "reference_date",
            "start_date",
            "end_date",
            "production_date",
            "polarization",
            "version",
        ]
        
        ccslc_df = pd.DataFrame(
            [Path(ix).stem.split("_") for ix in compressed_cslc], columns=columns
        )
        ccslc_df["file"] = [Path(ix).stem for ix in compressed_cslc]
        
        # Convert date columns
        for d in ["start_date", "end_date", "reference_date"]:
            ccslc_df[d] = pd.to_datetime(
                ccslc_df[d], format="%Y%m%dT%H%M%SZ", errors="raise"
            )
            ccslc_df[d] = ccslc_df[d].dt.date
        
        return ccslc_df

    @staticmethod
    def load_burst_db(url: str, frame_id: int) -> pd.DataFrame:
        """Load OPERA burst database JSON and filter by frame_id."""
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        
        logger = logging.getLogger(__name__)
        logger.info("Loading OPERA consistent burst_id database")
        logger.info(f"  Database generation date: {data['metadata']['generation_time']}")
        
        features = []
        for burst_id, info in data["data"].items():
            features.append({"frame_id": int(burst_id), **info})
        
        df = pd.DataFrame(features)
        return df[df.frame_id == frame_id]

    @staticmethod
    def normalize_db_id(db_id: str) -> str:
        """Convert DB format (t034_071052_iw1) to CSLC format (T034-071052-IW1)."""
        parts = db_id.split("_")
        track = parts[0].upper()
        frame = parts[1]
        iw = parts[2].upper()
        return f"{track}-{frame}-{iw}"

    def validate_cslc_coverage(self, cslc_df: pd.DataFrame) -> dict:
        """Validate CSLC coverage against burst database."""
        self.logger.info("Validating CSLC coverage against burst database")
        
        # Load and process burst database
        burst_db = self.load_burst_db(JSON_URL, self.frame_id)
        burst_db["burst_id_list_norm"] = burst_db["burst_id_list"].apply(
            lambda lst: [self.normalize_db_id(b) for b in lst]
        )
        
        db_norm_set = set(burst_db.burst_id_list_norm.values[0])
        
        # Analyze coverage by date
        result = (
            cslc_df.groupby("yyyymmdd")
            .agg(
                n_files=("burst_id", "size"),
                date_min=("date", "min"),
                date_max=("date", "max"),
                burst_ids=("burst_id", list),
                db_consist_flag=("burst_id", lambda x: all(b in db_norm_set for b in x)),
                missing_burst=("burst_id", lambda x: sorted(db_norm_set - set(x))),
                duplicated_burst=(
                    "burst_id",
                    lambda x: sorted(
                        x.value_counts()[x.value_counts() > 1].index.tolist()
                    ),
                ),
                unexpected_burst=("burst_id", lambda x: sorted(set(x) - db_norm_set)),
            )
            .reset_index()
        )
        
        self.logger.info(f"Runconfig contains {result.shape[0]} dates for frame {self.frame_id}")
        self.logger.info(f"Frame consists of {len(db_norm_set)} bursts")
        
        if result.db_consist_flag.all():
            self.logger.info("✓ All dates match burst database")
        else:
            self.logger.warning("⚠ Some dates have coverage issues:")
            for _, row in result.iterrows():
                issues = []
                if row["missing_burst"]:
                    issues.append(f"missing: {row['missing_burst']}")
                if row["duplicated_burst"]:
                    issues.append(f"duplicated: {row['duplicated_burst']}")
                if row["unexpected_burst"]:
                    issues.append(f"unexpected: {row['unexpected_burst']}")
                
                if issues:
                    self.logger.warning(f"  Date {row['yyyymmdd']}: {', '.join(issues)}")
        
        return result.to_dict()

    def download_ccslc_files(self, ccslc_list: list, max_workers: int = 5) -> list:
        """Download Compressed CSLC files from S3."""
        self.logger.info(f"Downloading {len(ccslc_list)} Compressed CSLC files")
        
        fs = s3fs.S3FileSystem(profile=self.aws_profile)
        bucket_prefix = "opera-ops-lts-pop1/products/CSLC_S1_COMPRESSED/"
        
        def download_file(key):
            local_path = self.ccslc_dir / f"{key.split('/')[-1]}.h5"
            if local_path.exists():
                self.logger.debug(f"Skipping (exists): {local_path.name}")
                return str(local_path)
            
            s3_path = f"{bucket_prefix}{key}/{key}.h5"
            self.logger.debug(f"Downloading {key}.h5")
            fs.get(s3_path, str(local_path))
            return str(local_path)
        
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_file, key) for key in ccslc_list]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    downloaded_files.append(result)
        
        self.logger.info(f"✓ Downloaded {len(downloaded_files)} Compressed CSLC files")
        return downloaded_files

    def set_aws_env(self, region: str = "us-west-2"):
        """Set AWS credentials from profile to environment for GDAL."""
        session = Session(profile=self.aws_profile)
        creds = session.get_credentials().get_frozen_credentials()
        
        gdal.SetConfigOption("AWS_REGION", region)
        gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", creds.secret_key)
        gdal.SetConfigOption("AWS_ACCESS_KEY_ID", creds.access_key)
        gdal.SetConfigOption("AWS_SESSION_TOKEN", creds.token)
        
        self.logger.debug(f"AWS credentials loaded from profile: {self.aws_profile}")

    def download_config_file(self, filename: str, base_url: str, fallback: str = None) -> Path:
        """Download a configuration file with fallback option."""
        local_path = self.config_dir / filename
        
        if local_path.exists():
            self.logger.debug(f"Config file exists: {filename}")
            return local_path
        
        try:
            url = base_url + filename
            self.logger.info(f"Downloading config: {filename}")
            r = requests.get(url)
            r.raise_for_status()
            local_path.write_bytes(r.content)
            return local_path
        except Exception as e:
            if fallback:
                self.logger.warning(f"Failed to download {filename}: {e}")
                self.logger.info(f"Falling back to: {fallback}")
                url = base_url + fallback
                r = requests.get(url)
                r.raise_for_status()
                fallback_path = self.config_dir / fallback
                fallback_path.write_bytes(r.content)
                return fallback_path
            else:
                raise

    def stage_all_data(self) -> dict:
        """Stage all required data for OPERA DISP-S1 processing."""
        self.logger.info("=" * 80)
        self.logger.info(f"Starting data staging for Frame {self.frame_id}")
        self.logger.info("=" * 80)
        
        staged_files = {}
        
        # Parse input files
        self.logger.info("\n[1/7] Parsing input files...")
        cslc_df = self._get_cslc_df(self.input_list)
        ccslc_df = self._get_ccslc_df(self.input_list)
        
        self.logger.info(f"  Found {len(cslc_df)} CSLC files")
        self.logger.info(f"  Found {len(ccslc_df)} Compressed CSLC files")
        
        # Validate coverage
        self.logger.info("\n[2/7] Validating CSLC coverage...")
        self.validate_cslc_coverage(cslc_df)
        
        # Download CSLC files
        self.logger.info("\n[3/7] Downloading CSLC files...")
        cslc_list = download_cslcs(
            burst_ids=cslc_df.burst_id.unique(),
            output_dir=self.cslc_dir,
            start=cslc_df.yyyymmdd.min() - pd.Timedelta(days=1),
            end=cslc_df.yyyymmdd.max() + pd.Timedelta(days=1),
        )
        staged_files["cslc"] = cslc_list
        self.logger.info(f"✓ Downloaded {len(cslc_list)} CSLC files")
        
        # Download Compressed CSLC files
        self.logger.info("\n[4/7] Downloading Compressed CSLC files...")
        ccslc_files = self.download_ccslc_files(ccslc_df.file.to_list())
        staged_files["ccslc"] = ccslc_files
        
        # Download static layers
        self.logger.info("\n[5/7] Downloading static layers...")
        static_lyrs = download_cslc_static_layers(
            cslc_df.burst_id.unique(), self.static_dir / "cslc"
        )
        staged_files["static_layers"] = static_lyrs
        self.logger.info(f"✓ Downloaded {len(static_lyrs)} static layer files")
        
        # Download DEM
        self.logger.info("\n[6/7] Downloading DEM...")
        self.set_aws_env()
        utm_epsg, utm_bounds = opera_utils.get_frame_bbox(frame_id=self.frame_id)
        bbox = opera_utils.reproject_bounds(utm_bounds, src_epsg=utm_epsg, dst_epsg=4326)
        dem_file = self.static_dir / "dem.vrt"
        
        if DISP_S1_INSTALLED:
            stage_dem(
                output=dem_file,
                bbox=bbox,
                margin=5,
                s3_bucket=S3_DEM_BUCKET,
                s3_key=S3_LONLAT_VRT_KEY,
            )
        else:
            self.logger.warning(
                "disp_s1 not installed, using basic DEM staging. "
                "For full functionality, install: pip install disp-s1"
            )
            self._stage_dem_basic(dem_file, bbox)
        
        staged_files["dem"] = str(dem_file)
        self.logger.info(f"✓ Created DEM: {dem_file.name}")
        
        # Create water mask
        self.logger.info("\n[7/7] Creating water mask...")
        water_file = self.static_dir / "water.tif"
        self._create_water_mask(water_file)
        staged_files["water"] = str(water_file)
        self.logger.info(f"✓ Created water mask: {water_file.name}")
        
        # Download ionosphere files
        self.logger.info("\n[8/8] Downloading ionosphere files...")
        
        if DISP_S1_INSTALLED:
            config = DownloadConfig(
                input_files=list(map(str, cslc_list)),
                output_dir=self.iono_dir,
                ionosphere_type=IonosphereType.JPLG,
                download_endpoint=DEFAULT_DOWNLOAD_ENDPOINT,
            )
            iono_files = download_ionosphere_files(config)
            staged_files["ionosphere"] = iono_files
            self.logger.info(f"✓ Downloaded {len(iono_files)} ionosphere files")
        else:
            self.logger.warning(
                "disp_s1 not installed, skipping ionosphere download. "
                "Install with: pip install disp-s1"
            )
            staged_files["ionosphere"] = []
        
        # Download configuration files
        self.logger.info("\n[9/9] Downloading configuration files...")
        self._download_config_files()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ Data staging complete!")
        self.logger.info("=" * 80)
        
        return staged_files

    def _stage_dem_basic(self, output_file: Path, bbox: tuple) -> None:
        """Stage basic DEM using GDAL directly (fallback when disp_s1 not installed)."""
        from osgeo import gdal
        
        x_min, y_min, x_max, y_max = bbox
        
        # Add margin (5 km approximately 0.045 degrees)
        margin = 0.045
        x_min -= margin
        y_min -= margin
        x_max += margin
        y_max += margin
        
        # Use GDAL to access S3 DEM
        s3_vrt = f"/vsis3/{S3_DEM_BUCKET}/{S3_LONLAT_VRT_KEY}"
        
        self.logger.info(f"Accessing DEM from: {s3_vrt}")
        
        # Open the source
        src_ds = gdal.Open(s3_vrt, gdal.GA_ReadOnly)
        if src_ds is None:
            raise RuntimeError(
                f"Failed to open DEM from S3. Check AWS credentials and network."
            )
        
        # Extract the region
        gdal.Warp(
            str(output_file),
            src_ds,
            format='VRT',
            outputBounds=[x_min, y_min, x_max, y_max],
            dstSRS='EPSG:4326',
        )
        
        src_ds = None
        self.logger.info(f"DEM VRT created: {output_file}")

    def _create_water_mask(self, output_file: Path):
        """Create water mask for the frame."""
        if create_water_mask is None:
            raise ImportError(
                "Water mask creation requires either the local water_mask.py module "
                "or the disp_s1 package. Please ensure water_mask.py is in the same "
                "directory as the script, or install disp_s1: pip install disp-s1"
            )
        
        create_water_mask(
            frame_id=self.frame_id,
            output=output_file,
            margin=5,
            land_buffer=1,
            ocean_buffer=1,
        )

    def _download_config_files(self):
        """Download all required configuration files."""
        runconfig = self.runconfig["RunConfig"]["Groups"][self.config_type]
        
        # Algorithm parameters
        try:
            alg_file = Path(
                runconfig["dynamic_ancillary_file_group"]["algorithm_parameters_file"]
            ).name
            self.download_config_file(alg_file, DISP_CONFIG_URL)
        except Exception as e:
            self.logger.warning(f"Using default algorithm parameters: {e}")
            self.download_config_file(
                "algorithm_parameters_historical_20251001.yaml", DISP_CONFIG_URL
            )
        
        # Frame to burst mapping
        try:
            frame_file = Path(
                runconfig["static_ancillary_file_group"]["frame_to_burst_json"]
            ).name
            version = frame_file.split("-")[3]
            self.download_config_file(frame_file, f"{BURST_DB_URL}v{version}/")
        except Exception as e:
            self.logger.warning(f"Using default frame-to-burst mapping: {e}")
            self.download_config_file(
                "opera-s1-disp-0.13.0-frame-to-burst.json.zip",
                f"{BURST_DB_URL}v0.13.0/",
            )
        
        # Reference date database
        try:
            ref_file = Path(
                runconfig["static_ancillary_file_group"]["reference_date_database_json"]
            ).name
            self.download_config_file(ref_file, DISP_CONFIG_URL + "static_ancillary_files/")
        except Exception as e:
            self.logger.warning(f"Using default reference date database: {e}")
            self.download_config_file(
                "opera-disp-s1-reference-dates-2025-02-13.json",
                DISP_CONFIG_URL + "static_ancillary_files/",
            )
        
        # Algorithm overrides
        try:
            override_file = Path(
                runconfig["static_ancillary_file_group"][
                    "algorithm_parameters_overrides_json"
                ]
            ).name
            self.download_config_file(override_file, DISP_CONFIG_URL)
        except Exception as e:
            self.logger.warning(f"Using default algorithm overrides: {e}")
            self.download_config_file(
                "opera-disp-s1-algorithm-parameters-overrides-2025-09-17.json",
                DISP_CONFIG_URL,
            )

    def create_updated_runconfig(self, staged_files: dict) -> Path:
        """Create updated runconfig with absolute local file paths."""
        self.logger.info("Creating updated runconfig...")
        
        # Combine CSLC and CCSLC file lists - convert to absolute paths
        input_list = [str(Path(f).resolve()) for f in staged_files["cslc"]]
        input_list.extend([str(Path(f).resolve()) for f in staged_files["ccslc"]])
        
        # Get config file paths
        config_files = list(self.config_dir.glob("*"))
        alg_file = next(
            (f for f in config_files if "algorithm_parameters" in f.name and "override" not in f.name),
            None,
        )
        frame_file = next((f for f in config_files if "frame-to-burst" in f.name), None)
        ref_file = next((f for f in config_files if "reference-dates" in f.name), None)
        override_file = next(
            (f for f in config_files if "overrides" in f.name), None
        )
        
        # Update runconfig with absolute paths
        runconfig = self.runconfig["RunConfig"]["Groups"][self.config_type]
        runconfig["input_file_group"]["cslc_file_list"] = input_list
        runconfig["dynamic_ancillary_file_group"]["algorithm_parameters_file"] = (
            str(alg_file.resolve()) if alg_file else ""
        )
        runconfig["dynamic_ancillary_file_group"]["static_layers_files"] = [
            str(Path(f).resolve()) for f in staged_files["static_layers"]
        ]
        runconfig["dynamic_ancillary_file_group"]["mask_file"] = str(
            Path(staged_files["water"]).resolve()
        )
        runconfig["dynamic_ancillary_file_group"]["dem_file"] = str(
            Path(staged_files["dem"]).resolve()
        )
        runconfig["dynamic_ancillary_file_group"]["ionosphere_files"] = [
            str(Path(f).resolve()) for f in staged_files["ionosphere"]
        ]
        runconfig["static_ancillary_file_group"]["frame_to_burst_json"] = (
            str(frame_file.resolve()) if frame_file else ""
        )
        runconfig["static_ancillary_file_group"]["reference_date_database_json"] = (
            str(ref_file.resolve()) if ref_file else ""
        )
        runconfig["static_ancillary_file_group"][
            "algorithm_parameters_overrides_json"
        ] = (str(override_file.resolve()) if override_file else "")
        
        # Update output paths with absolute paths
        runconfig["product_path_group"]["product_path"] = str(
            (self.frame_dir / "output_dir").resolve()
        )
        runconfig["product_path_group"]["scratch_path"] = str(
            (self.frame_dir / "scratch_path").resolve()
        )
        runconfig["product_path_group"]["sas_output_path"] = str(
            (self.frame_dir / "output_dir").resolve()
        )
        runconfig["log_file"] = str((self.frame_dir / "disp-s1-sas.log").resolve())
        
        # Write updated runconfig
        output_runconfig = self.frame_dir / "Runconfig.yaml"
        with open(output_runconfig, "w") as f:
            yaml.dump(runconfig, f, sort_keys=False)
        
        self.logger.info(f"✓ Updated runconfig saved to: {output_runconfig.resolve()}")
        return output_runconfig


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Stage OPERA DISP-S1 processing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage data with default settings
  %(prog)s runconfig.yaml -o /data/staging
  
  # Use custom AWS profile
  %(prog)s runconfig.yaml -o /data/staging --aws-profile my-profile
  
  # Enable debug logging
  %(prog)s runconfig.yaml -o /data/staging --log-level DEBUG
        """,
    )
    
    parser.add_argument(
        "runconfig",
        type=Path,
        help="Path to input runconfig YAML file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for staged data",
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default="saml-pub",
        help="AWS profile name (default: saml-pub)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Validate inputs
    if not args.runconfig.exists():
        logger.error(f"Runconfig file not found: {args.runconfig}")
        sys.exit(1)
    
    try:
        # Initialize stager
        stager = DataStager(
            runconfig_path=args.runconfig,
            output_dir=args.output_dir,
            aws_profile=args.aws_profile,
        )
        
        # Stage all data
        staged_files = stager.stage_all_data()
        
        # Create updated runconfig
        updated_runconfig = stager.create_updated_runconfig(staged_files)
        
        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS!")
        logger.info(f"Updated runconfig: {updated_runconfig}")
        logger.info(f"Staged data location: {stager.frame_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during data staging: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()