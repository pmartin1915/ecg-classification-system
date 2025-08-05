"""
Non-interactive cleanup script - performs the actual cleanup
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
import zipfile

def get_redundant_files():
    """Get list of files to remove"""
    return {
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
        'app/components/dnp_education_module.py',
        
        # Redundant Utils
        'app/utils/comprehensive_mapper.py',
        'app/utils/optimized_dataset_loader.py',
        'app/utils/ecg_generator.py',
        
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
        'system_enhancement_analysis.py',
        
        # Legacy Test/Demo Files
        'clean_demo.py',
        'proof_of_concept_demo.py',
        'minimal_test.py',
        'simple_test.py',
        'quick_test.py',
        'simple_test_phase2.py',
        'test_mi_quick.py',
        'working_combined_test.py',
        'test_combined_dataset.py',
        'test_comprehensive_cardiac.py',
        
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
        'simple_validation.py',
        'validate_integration.py',
        
        # Redundant Dataset Tools
        'fix_arrhythmia_loader.py',
        'optimized_arrhythmia_loader.py',
        
        # Redundant Launch Scripts
        'launch_comprehensive_system.py',
        'quick_launch_comprehensive.py',
        'run_full_dataset_analysis.py',
        
        # Fix/Maintenance Scripts
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
        'setup.py'
    }

def create_backup(backup_dir: Path, files_to_backup: set):
    """Create backup of files to be removed"""
    print(f"Creating backup directory: {backup_dir}")
    backup_dir.mkdir(exist_ok=True)
    
    # Create timestamped zip backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_zip = backup_dir / f"removed_files_backup_{timestamp}.zip"
    
    backed_up_count = 0
    
    with zipfile.ZipFile(backup_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                zipf.write(file_path, file_path)
                backed_up_count += 1
                print(f"  [BACKED UP] {file_path}")
    
    print(f"\nBACKUP CREATED: {backup_zip}")
    print(f"Files backed up: {backed_up_count}")
    return backup_zip, backed_up_count

def remove_files(files_to_remove: set):
    """Remove the redundant files"""
    removed_count = 0
    not_found_count = 0
    
    print(f"\nREMOVING REDUNDANT FILES:")
    print("=" * 50)
    
    for file_path in sorted(files_to_remove):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  [REMOVED] {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"  [ERROR] Could not remove {file_path}: {e}")
        else:
            print(f"  [NOT FOUND] {file_path}")
            not_found_count += 1
    
    print("=" * 50)
    print(f"Files removed: {removed_count}")
    print(f"Files not found: {not_found_count}")
    
    return removed_count

def main():
    """Perform the cleanup"""
    print("ECG PROJECT CLEANUP - PERFORMING CLEANUP")
    print("=" * 60)
    
    # Get files to remove
    redundant_files = get_redundant_files()
    
    print(f"Files to be removed: {len(redundant_files)}")
    print(f"Creating backup and removing files...")
    
    # Create backup
    backup_dir = Path('backup_removed_files')
    backup_zip, backed_up_count = create_backup(backup_dir, redundant_files)
    
    # Remove files
    removed_count = remove_files(redundant_files)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("CLEANUP COMPLETED!")
    print("=" * 60)
    print(f"[BACKUP] {backed_up_count} files backed up to: {backup_zip}")
    print(f"[REMOVED] {removed_count} redundant files removed")
    print(f"[RESULT] Project cleaned up - {removed_count} fewer files!")
    
    # Show what's left
    remaining_py = len(list(Path('.').rglob('*.py')))
    remaining_bat = len(list(Path('.').rglob('*.bat')))
    
    print(f"\nREMAINING FILES:")
    print(f"Python files: {remaining_py}")
    print(f"Batch files: {remaining_bat}")
    print(f"Total: {remaining_py + remaining_bat}")
    
    return True

if __name__ == "__main__":
    success = main()