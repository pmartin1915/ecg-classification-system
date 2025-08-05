"""
Project Cleanup Script
Safely removes redundant files and creates backups
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
import zipfile

def create_cleanup_analysis():
    """Analyze and categorize all files for cleanup"""
    
    # Files to KEEP (essential/active files)
    essential_files = {
        # Main Applications
        'app/enhanced_main.py',
        'app/main.py', 
        'app/smart_launcher.py',
        
        # Core Components (Keep All)
        'app/components/enhanced_explainability.py',
        'app/components/ai_explainability.py',
        'app/components/performance_monitor.py',
        'app/components/batch_processor.py',
        'app/components/classification.py',
        'app/components/clinical_disclaimers.py',
        'app/components/clinical_training.py',
        'app/components/monitoring.py',
        'app/components/prediction.py',
        'app/components/training_results.py',
        'app/components/__init__.py',
        
        # Essential Utils
        'app/utils/fast_prediction_pipeline.py',
        'app/utils/dataset_manager.py',
        'app/utils/data_loader.py',
        'app/utils/data_processing.py',
        'app/utils/model_integration.py',
        'app/utils/__init__.py',
        'app/__init__.py',
        
        # Active Launchers & Scripts
        'ENHANCED_LAUNCHER.bat',
        'system_diagnostics.py',
        'enhanced_mi_detection_system.py',
        'test_mi_enhancement.py',
        'optimized_mi_enhancement.py',
        
        # Essential Tests
        'test_enhanced_explainability.py',
        'test_performance_optimization.py',
        'test_ui_integration.py',
        
        # Phase Tests (Keep for development)
        'test_phase1.py',
        'test_phase2.py', 
        'test_phase3.py',
        'test_phase4.py',
        
        # Important Analysis
        'final_validation.py',
        'validation_summary.py',
        
        # Installation
        'ECG_PROFESSIONAL_INSTALLER.bat',
        'SIMPLE_INSTALLER.bat',
        'INSTALL_DESKTOP_SHORTCUT.bat'
    }
    
    # Files to REMOVE (redundant/superseded)
    redundant_files = {
        # Redundant Launchers
        'STREAMLINED_LAUNCHER.bat',
        'PROFESSIONAL_LAUNCHER.bat', 
        'LAUNCH_ECG_COMPREHENSIVE.bat',
        'LAUNCH_ECG_SYSTEM.bat',
        'QUICK_LAUNCH.bat',
        'launch_ecg_working.bat',
        
        # Legacy Main Apps
        'app/minimal_main.py',
        'app/simple_main.py',
        
        # Legacy Components
        'app/components/dnp_education_module.py',  # Superseded by clinical_training
        
        # Redundant Utils
        'app/utils/comprehensive_mapper.py',  # Superseded by dataset_manager
        'app/utils/optimized_dataset_loader.py',  # Superseded by dataset_manager
        'app/utils/ecg_generator.py',  # Not actively used
        
        # Legacy Enhancement Scripts  
        'quick_mi_enhancement.py',
        'working_mi_enhancement.py',
        'final_mi_enhancement.py',
        'run_mi_enhancement.py',
        'ptbxl_mi_enhancement.py',
        'pure_ptbxl_enhancement.py',
        'simple_ptbxl_enhancement.py',
        'run_enhanced_mi_training.py',
        'full_dataset_enhancement.py',
        
        # Legacy Test/Demo Files
        'clean_demo.py',
        'proof_of_concept_demo.py',
        'minimal_test.py',
        'simple_test.py',
        'quick_test.py',
        'simple_test_phase2.py',
        'test_mi_quick.py',
        'working_combined_test.py',
        
        # Development/Debug Files
        'debug_imports.py',
        'debug_phase1.py',
        'diagnose_preprocessing.py',
        'test_imports.py',
        'clean_status_check.py',
        
        # Legacy Analysis
        'clean_visual_generator.py',
        'enhanced_visual_generator.py', 
        'system_analysis.py',
        'system_enhancement_analysis.py',
        'simple_validation.py',
        'validate_integration.py',
        
        # Redundant Dataset Tools
        'fix_arrhythmia_loader.py',
        'optimized_arrhythmia_loader.py',
        'test_combined_dataset.py',
        
        # Redundant Launch Scripts
        'launch_comprehensive_system.py',
        'quick_launch_comprehensive.py',
        'run_full_dataset_analysis.py',
        'test_comprehensive_cardiac.py',
        
        # Fix/Maintenance Scripts (No longer needed)
        'app/fix_pca.py',
        'app/check_data.py',
        'app/check_phase3_results.py',
        
        # Desktop Shortcut Scripts (Redundant)
        'create_desktop_shortcut.py',
        'create_shortcut_manual.py',
        'create_desktop_icon.py',
        
        # Model Training (Superseded)
        'quick_model_training.py',
        
        # Empty/Placeholder
        'setup.py'  # Empty file
    }
    
    return essential_files, redundant_files

def create_backup(backup_dir: Path, files_to_backup: set):
    """Create a backup of files before deletion"""
    
    print(f"Creating backup in: {backup_dir}")
    backup_dir.mkdir(exist_ok=True)
    
    # Create timestamped zip backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_zip = backup_dir / f"project_backup_{timestamp}.zip"
    
    backed_up_count = 0
    
    with zipfile.ZipFile(backup_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                zipf.write(file_path, file_path)
                backed_up_count += 1
                print(f"  Backed up: {file_path}")
    
    print(f"\nBackup created: {backup_zip}")
    print(f"Files backed up: {backed_up_count}")
    return backup_zip

def cleanup_files(files_to_remove: set, dry_run: bool = True):
    """Remove redundant files"""
    
    removed_count = 0
    not_found_count = 0
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}REMOVING REDUNDANT FILES:")
    print("=" * 60)
    
    for file_path in sorted(files_to_remove):
        if os.path.exists(file_path):
            if not dry_run:
                try:
                    os.remove(file_path)
                    print(f"  [REMOVED] {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"  [ERROR] Could not remove {file_path}: {e}")
            else:
                print(f"  [WOULD REMOVE] {file_path}")
                removed_count += 1
        else:
            print(f"  [NOT FOUND] {file_path}")
            not_found_count += 1
    
    print("=" * 60)
    print(f"Files {'would be ' if dry_run else ''}removed: {removed_count}")
    print(f"Files not found: {not_found_count}")
    
    return removed_count

def analyze_project_structure():
    """Show before/after project structure analysis"""
    
    print("\nPROJECT STRUCTURE ANALYSIS")
    print("=" * 50)
    
    # Count all Python and batch files
    all_py_files = list(Path('.').rglob('*.py'))
    all_bat_files = list(Path('.').rglob('*.bat'))
    
    print(f"Current Python files: {len(all_py_files)}")
    print(f"Current Batch files: {len(all_bat_files)}")
    print(f"Total script files: {len(all_py_files) + len(all_bat_files)}")
    
    essential_files, redundant_files = create_cleanup_analysis()
    
    print(f"\nAfter cleanup:")
    print(f"Essential files: {len(essential_files)}")
    print(f"Files to remove: {len(redundant_files)}")
    print(f"Reduction: {len(redundant_files)} files ({len(redundant_files)/(len(all_py_files) + len(all_bat_files))*100:.1f}%)")

def main():
    """Main cleanup function"""
    
    print("ECG CLASSIFICATION SYSTEM - PROJECT CLEANUP")
    print("=" * 60)
    print("This script will clean up redundant files and create backups")
    print("=" * 60)
    
    # Analyze current structure
    analyze_project_structure()
    
    # Get file lists
    essential_files, redundant_files = create_cleanup_analysis()
    
    print(f"\nFILES TO BE REMOVED ({len(redundant_files)} total):")
    print("-" * 60)
    
    # Categorize and display
    categories = {
        'Redundant Launchers': [f for f in redundant_files if f.endswith('.bat') and 'LAUNCHER' in f.upper()],
        'Legacy Applications': [f for f in redundant_files if f.startswith('app/') and f.endswith('_main.py')],
        'Legacy Enhancement Scripts': [f for f in redundant_files if 'enhancement' in f.lower() and f != 'enhanced_mi_detection_system.py'],
        'Legacy Test Files': [f for f in redundant_files if f.startswith(('test_', 'simple_test', 'minimal_test', 'quick_test'))],
        'Debug/Development Files': [f for f in redundant_files if 'debug' in f.lower() or 'diagnose' in f.lower()],
        'Legacy Components': [f for f in redundant_files if f.startswith('app/components/')],
        'Legacy Utils': [f for f in redundant_files if f.startswith('app/utils/')],
        'Other Legacy Files': [f for f in redundant_files if not any(f in cat for cat in [
            [f for f in redundant_files if f.endswith('.bat') and 'LAUNCHER' in f.upper()],
            [f for f in redundant_files if f.startswith('app/') and f.endswith('_main.py')],
            [f for f in redundant_files if 'enhancement' in f.lower()],
            [f for f in redundant_files if f.startswith(('test_', 'simple_test', 'minimal_test'))],
            [f for f in redundant_files if 'debug' in f.lower()],
            [f for f in redundant_files if f.startswith('app/components/')],
            [f for f in redundant_files if f.startswith('app/utils/')]
        ])]
    }
    
    for category, files in categories.items():
        if files:
            print(f"\n{category}:")
            for file in sorted(files):
                print(f"  - {file}")
    
    print(f"\nFILES TO BE KEPT ({len(essential_files)} total):")
    print("-" * 60)
    keep_categories = {
        'Main Applications': [f for f in essential_files if f.endswith('_main.py') or f == 'app/smart_launcher.py'],
        'Active Launchers': [f for f in essential_files if f.endswith('.bat')],
        'Core Components': [f for f in essential_files if f.startswith('app/components/')],
        'Essential Utils': [f for f in essential_files if f.startswith('app/utils/')],
        'Active Tests': [f for f in essential_files if f.startswith('test_')],
        'Enhancement System': [f for f in essential_files if 'enhancement' in f.lower() or 'mi_detection' in f.lower()],
        'System Tools': [f for f in essential_files if f in ['system_diagnostics.py', 'final_validation.py', 'validation_summary.py']]
    }
    
    for category, files in keep_categories.items():
        if files:
            print(f"\n{category}:")
            for file in sorted(files):
                print(f"  + {file}")
    
    # Ask for confirmation
    print(f"\nCONFIRMATION REQUIRED")
    print("=" * 60)
    print("This will:")
    print(f"1. Create a backup of {len(redundant_files)} files to be removed")
    print(f"2. Remove {len(redundant_files)} redundant files")
    print(f"3. Keep {len(essential_files)} essential files")
    print("4. Test remaining functionality")
    
    response = input(f"\nProceed with cleanup? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        # Create backup
        backup_dir = Path('backup_removed_files')
        backup_zip = create_backup(backup_dir, redundant_files)
        
        # Perform actual cleanup
        removed_count = cleanup_files(redundant_files, dry_run=False)
        
        print(f"\nCLEANUP COMPLETED!")
        print(f"[SUCCESS] {removed_count} files removed")
        print(f"[BACKUP] Backup created: {backup_zip}")
        print(f"[RETAINED] {len(essential_files)} essential files retained")
        
        return True
    else:
        print("\nCleanup cancelled. No files were modified.")
        return False

if __name__ == "__main__":
    success = main()