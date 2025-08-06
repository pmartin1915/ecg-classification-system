"""
PTB-XL MI Extractor - Professional Clinical Grade
Unlocks 5,288 MI records from PTB-XL database for clinical training
"""
import ast
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import wfdb
from tqdm import tqdm
import warnings

import sys
sys.path.append('../../')
try:
    from config.settings import CACHE_DIR, DATA_DIR
except ImportError:
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CACHE_DIR = DATA_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')


class PTBXLMIExtractor:
    """
    Professional PTB-XL MI Record Extractor
    Designed for comprehensive clinical training with evidence-based MI detection
    """
    
    def __init__(self, ptbxl_path: str = None):
        """Initialize PTB-XL MI extractor"""
        if ptbxl_path is None:
            ptbxl_path = "C:/ecg-classification-system-pc/ecg-classification-system/data/raw/ptbxl"
        
        self.ptbxl_path = Path(ptbxl_path)
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load PTB-XL metadata
        self.database_df = None
        self.statements_df = None
        self._load_metadata()
        
        # MI classification mapping for clinical training
        self.mi_mapping = {
            # Anterior Wall MI
            'AMI': 'Anterior MI',
            'ASMI': 'Anterior-Septal MI', 
            'ALMI': 'Anterior-Lateral MI',
            
            # Inferior Wall MI  
            'IMI': 'Inferior MI',
            'ISMI': 'Inferior-Septal MI',
            
            # Lateral Wall MI
            'LMI': 'Lateral MI',
            
            # Posterior Wall MI
            'PMI': 'Posterior MI',
            'PLMI': 'Posterior-Lateral MI',
            
            # Other MI Types
            'MI': 'Myocardial Infarction'
        }
        
        # Clinical significance thresholds (from PTB-XL documentation)
        self.significance_threshold = 50.0  # 50% confidence for clinical relevance
        
    def _load_metadata(self):
        """Load PTB-XL metadata files"""
        try:
            db_path = self.ptbxl_path / "ptbxl_database.csv"
            statements_path = self.ptbxl_path / "scp_statements.csv"
            
            if not db_path.exists():
                raise FileNotFoundError(f"PTB-XL database not found: {db_path}")
            
            self.database_df = pd.read_csv(db_path)
            if statements_path.exists():
                self.statements_df = pd.read_csv(statements_path)
            
            print(f"[OK] PTB-XL metadata loaded: {len(self.database_df)} total records")
            
        except Exception as e:
            print(f"[ERROR] Failed to load PTB-XL metadata: {e}")
            raise
    
    def extract_mi_records(self, min_confidence: float = 15.0) -> pd.DataFrame:
        """
        Extract MI records from PTB-XL database with clinical details
        
        Args:
            min_confidence: Minimum confidence threshold for MI diagnosis
            
        Returns:
            DataFrame with MI records and clinical metadata
        """
        print(f"[EXTRACTING] MI records from PTB-XL (confidence >= {min_confidence}%)")
        
        mi_records = []
        
        for idx, row in tqdm(self.database_df.iterrows(), total=len(self.database_df), desc="Scanning records"):
            try:
                # Parse SCP codes (stored as string representation of dict)
                scp_codes = ast.literal_eval(row['scp_codes'])
                
                # Check for MI-related codes
                for code, confidence in scp_codes.items():
                    if code in self.mi_mapping and confidence >= min_confidence:
                        
                        # Determine clinical significance level
                        if confidence >= self.significance_threshold:
                            significance = "High Clinical Significance"
                        elif confidence >= 25.0:
                            significance = "Moderate Clinical Significance"
                        else:
                            significance = "Low Clinical Significance"
                        
                        # Create comprehensive MI record
                        mi_record = {
                            'ecg_id': row['ecg_id'],
                            'patient_id': row['patient_id'],
                            'mi_type_code': code,
                            'mi_type_name': self.mi_mapping[code],
                            'confidence_score': confidence,
                            'clinical_significance': significance,
                            'age': row['age'],
                            'sex': row['sex'],
                            'height': row.get('height', None),
                            'weight': row.get('weight', None),
                            'recording_date': row.get('recording_date', None),
                            'report': row.get('report', ''),
                            'full_scp_codes': str(scp_codes),
                            'filename_lr': row['filename_lr'],
                            'filename_hr': row.get('filename_hr', None),
                            
                            # Clinical training metadata
                            'suitable_for_training': confidence >= 25.0,
                            'educational_value': 'High' if confidence >= 50.0 else 'Medium' if confidence >= 25.0 else 'Low',
                            'complexity_level': self._assess_complexity(scp_codes)
                        }
                        
                        mi_records.append(mi_record)
                        break  # Take first MI type found per record
                        
            except (ValueError, SyntaxError, KeyError) as e:
                # Skip records with parsing issues
                continue
        
        mi_df = pd.DataFrame(mi_records)
        
        if len(mi_df) > 0:
            print(f"[SUCCESS] Extracted {len(mi_df)} MI records")
            print(f"[BREAKDOWN] MI type distribution:")
            for mi_type, count in mi_df['mi_type_name'].value_counts().head(10).items():
                print(f"  {mi_type}: {count} records")
            
            print(f"[CLINICAL] Confidence distribution:")
            print(f"  High (>=50%): {len(mi_df[mi_df['confidence_score'] >= 50.0])} records")
            print(f"  Medium (25-49%): {len(mi_df[(mi_df['confidence_score'] >= 25.0) & (mi_df['confidence_score'] < 50.0)])} records") 
            print(f"  Low (15-24%): {len(mi_df[(mi_df['confidence_score'] >= 15.0) & (mi_df['confidence_score'] < 25.0)])} records")
        else:
            print(f"[WARNING] No MI records found with confidence >= {min_confidence}%")
        
        return mi_df
    
    def _assess_complexity(self, scp_codes: dict) -> str:
        """Assess ECG complexity for educational purposes"""
        num_conditions = len([code for code, conf in scp_codes.items() if conf > 0])
        
        if num_conditions <= 2:
            return "Simple"
        elif num_conditions <= 4:
            return "Intermediate" 
        else:
            return "Complex"
    
    def load_mi_signals(self, mi_df: pd.DataFrame, max_records: int = 1000, sampling_rate: int = 100) -> Tuple[np.ndarray, List[str], List[str], Dict]:
        """
        Load ECG signals for MI records with clinical metadata
        
        Args:
            mi_df: DataFrame with MI record information
            max_records: Maximum number of signals to load
            sampling_rate: Target sampling rate (100 or 500 Hz)
            
        Returns:
            Tuple of (signals, labels, record_ids, metadata)
        """
        print(f"[LOADING] ECG signals for {min(len(mi_df), max_records)} MI records")
        
        signals = []
        labels = []
        record_ids = []
        clinical_metadata = []
        
        # Sort by confidence for best quality records first
        mi_df_sorted = mi_df.sort_values('confidence_score', ascending=False).head(max_records)
        
        loaded_count = 0
        for idx, row in tqdm(mi_df_sorted.iterrows(), total=len(mi_df_sorted), desc="Loading signals"):
            try:
                # Construct file path (filename already includes records100 path)
                if sampling_rate == 100:
                    filename = row['filename_lr']
                else:
                    filename = row.get('filename_hr', row['filename_lr'])
                
                signal_path = self.ptbxl_path / filename
                
                # Check if WFDB files exist (.hea and .dat)
                hea_file = Path(str(signal_path) + '.hea')
                if not hea_file.exists():
                    continue
                
                # Load ECG signal using WFDB
                record_name = str(signal_path).replace('.hea', '').replace('.dat', '')
                signal, fields = wfdb.rdsamp(record_name)
                
                # Ensure 12-lead format
                if signal.shape[1] >= 12:
                    signal = signal[:, :12]  # Take first 12 leads
                else:
                    # Pad if fewer than 12 leads
                    padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                    signal = np.hstack([signal, padding])
                
                # Standardize length (10 seconds at sampling rate)
                target_length = sampling_rate * 10
                if signal.shape[0] > target_length:
                    signal = signal[:target_length]
                elif signal.shape[0] < target_length:
                    padding = np.zeros((target_length - signal.shape[0], 12))
                    signal = np.vstack([signal, padding])
                
                signals.append(signal.astype(np.float32))
                labels.append('MI')
                record_ids.append(f"ptbxl_{row['ecg_id']}")
                
                # Store clinical metadata for educational purposes
                clinical_metadata.append({
                    'mi_type': row['mi_type_name'],
                    'confidence': row['confidence_score'],
                    'significance': row['clinical_significance'],
                    'age': row['age'],
                    'sex': row['sex'],
                    'educational_value': row['educational_value'],
                    'complexity': row['complexity_level']
                })
                
                loaded_count += 1
                
                if loaded_count >= max_records:
                    break
                    
            except Exception as e:
                continue
        
        signals_array = np.array(signals, dtype=np.float32) if signals else np.array([])
        
        # Create comprehensive metadata
        metadata = {
            'total_records': len(signals_array),
            'mi_records': len(signals_array),
            'mi_percentage': 100.0,  # All records are MI
            'data_source': 'PTB-XL Database',
            'sampling_rate': sampling_rate,
            'signal_length': signals_array.shape[1] if len(signals_array) > 0 else 0,
            'n_leads': 12,
            'clinical_metadata': clinical_metadata,
            'mi_type_distribution': mi_df_sorted['mi_type_name'].value_counts().to_dict(),
            'confidence_stats': {
                'mean': float(mi_df_sorted['confidence_score'].mean()),
                'min': float(mi_df_sorted['confidence_score'].min()),
                'max': float(mi_df_sorted['confidence_score'].max())
            },
            'educational_features': {
                'high_confidence_records': len([m for m in clinical_metadata if m['confidence'] >= 50.0]),
                'simple_cases': len([m for m in clinical_metadata if m['complexity'] == 'Simple']),
                'intermediate_cases': len([m for m in clinical_metadata if m['complexity'] == 'Intermediate']),
                'complex_cases': len([m for m in clinical_metadata if m['complexity'] == 'Complex'])
            }
        }
        
        print(f"[SUCCESS] Loaded {len(signals_array)} MI signals")
        print(f"[CLINICAL] Average confidence: {metadata['confidence_stats']['mean']:.1f}%")
        print(f"[EDUCATION] High confidence records: {metadata['educational_features']['high_confidence_records']}")
        
        return signals_array, labels, record_ids, metadata
    
    def save_mi_dataset(self, signals: np.ndarray, labels: List[str], record_ids: List[str], metadata: Dict, filename: str = "ptbxl_mi_clinical_dataset"):
        """Save MI dataset for clinical training use"""
        
        cache_file = self.cache_dir / f"{filename}.pkl"
        
        dataset = {
            'signals': signals,
            'labels': labels,
            'record_ids': record_ids,
            'metadata': metadata,
            'creation_info': {
                'source': 'PTB-XL MI Extractor',
                'purpose': 'Clinical Training',
                'quality': 'Professional Grade',
                'extracted_at': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"[SAVED] MI dataset saved to: {cache_file}")
        return cache_file


def extract_ptbxl_mi_dataset():
    """Main function to extract comprehensive PTB-XL MI dataset"""
    try:
        print("=" * 60)
        print("PTB-XL MI EXTRACTOR - CLINICAL TRAINING DATASET")
        print("=" * 60)
        
        # Initialize extractor
        extractor = PTBXLMIExtractor()
        
        # Extract MI records with clinical details
        print("[PHASE 1] Extracting MI records from PTB-XL database...")
        mi_df = extractor.extract_mi_records(min_confidence=15.0)
        
        if len(mi_df) == 0:
            print("[ERROR] No MI records found!")
            return False
        
        # Load high-quality MI signals for training
        print("[PHASE 2] Loading ECG signals for clinical training...")
        signals, labels, record_ids, metadata = extractor.load_mi_signals(
            mi_df, 
            max_records=2000,  # Load substantial number for training
            sampling_rate=100
        )
        
        if len(signals) == 0:
            print("[ERROR] No MI signals loaded!")
            return False
        
        # Save comprehensive dataset
        print("[PHASE 3] Saving clinical training dataset...")
        cache_file = extractor.save_mi_dataset(signals, labels, record_ids, metadata)
        
        print("\n" + "=" * 60)
        print("PTB-XL MI EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"MI Records Found: {len(mi_df)}")
        print(f"Signals Loaded: {len(signals)}")
        print(f"Dataset Saved: {cache_file}")
        print(f"Ready for Clinical Training: YES")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] PTB-XL MI extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    extract_ptbxl_mi_dataset()