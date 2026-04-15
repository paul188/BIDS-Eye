import yaml
import re

def clean_suffixes(data_string):
    data = yaml.safe_load(data_string)
    
    if 'suffix' not in data:
        return data

    cleaned_suffix = {}
    
    # 1. Broad categories of noise/system strings
    system_noise = {
        '00index', '0a', '0b', 'log', 'logfile', 'metadata', 'notes', 
        'report', 'script', 'scripts', 'summary', 'metadata', 'json', 'matlab'
    }
    
    # Behavioral tasks, movie titles, and game names
    task_noise = {
        'theblueumbrella', 'toystory2', 'walle', 'zelda', 'up', 'target', 
        'taskswitch', 'test', 'tests', 'usage', 'video', 'visualacuity',
        'vocab', 'verbs', 'watching', 'work', 'usage', 'table', 'variables', 
        'values', 'textmaps', 'toolbox', 'total', 'laluna', 'lemmings', 
        'lifted', 'partlycloudy', 'pinky', 'piper', 'plaqueattack', 'presto', 
        'sintel', 'wheretheressmoke', 'wind', 'language', 'metaphors', 
        'phrases', 'photo', 'recognition', 'saccades', 'sokoban', 'spanish', 
        'thumb', 'recall', 'lastcall', 'interest', 'job'
    }

    # Clinical scales, inventories, and psychometric tests
    scale_noise = {
        'wais', 'wasi', 'whodas', 'ymrs', 'tipi', 'towre', 'wraml', 'wrat3', 
        'upps', 'vas', 'vmnm', 'vcap', 'ksads', 'madrs', 'masq', 'mcas', 
        'mfq', 'mfqc', 'mini', 'mpq', 'mspss01', 'musebaq', 'nffi01', 
        'panas', 'panss01', 'pdss01', 'phq8', 'piat', 'psqi', 'pss', 
        'qids01', 'question', 'questionnaire', 'survey', 'iri', 'irq', 
        'kbit', 'lifestyle', 'rt01', 'sans', 'saps', 'satisfaction', 
        'scap', 'scid', 'shaps01', 'sils01', 'stai', 'staic', 'stroop', 
        'ymrs01', 'kbit'
    }

    # Directional / Anatomical labels often mislabeled as suffixes
    directional_noise = {
        'left', 'right', 'medial', 'lateral', 'ventral', 'outer', 'middle', 
        'inner', 'top', 'bottom', 'orientation'
    }

    for k, v in data['suffix'].items():
        k_str = str(k)
        k_lower = k_str.lower()

        # --- A. NUMERIC / ENUMERATOR FILTERS ---

        # Delete pure numbers
        try:
            float(k_str)
            continue 
        except (ValueError, TypeError):
            pass

        # Delete Volume, Timepoint, Subject, level, and picture indices
        # We catch t3-t9 but preserve t1 and t2 (common shorthand)
        if re.search(r'^t[3-9]$|^t\d{2,}$', k_lower): continue 
        if re.search(r'^vol\d+$|^v\d{3,}$', k_lower): continue 
        if re.search(r'^sub\d+$|^qb\d+$|^s\d+$|^lvl\d+$|^pict\d+$', k_lower): continue

        # --- B. PATTERN / DERIVATIVE FILTERS ---

        # Delete ANTs / Registration / Transformation derivatives
        if any(x in k_str for x in ['Warp', 'Affine', 'Inverse', 'Xfms', 'Reg', 'registration']):
            continue

        # Delete Resolution markers
        if re.search(r'\d+um$', k_lower): continue

        # Delete Statistical / Metric / Metric outputs
        # Clears cope, zstat, tstat, mean, kurt, skew, metrics, etc.
        stat_patterns = ['stat', 'thresh', 'cope', 'tsnr', 'tmask', 'vpset', 'vpdat', 'metrics', 'prob', 'mean', 'mask', 'skew', 'kurt', 'stdev', 'mcf', 'moco', 'moma', 'nu', 'norm', 'orig']
        if any(p in k_lower for p in stat_patterns):
            continue

        # Delete Surface / Segmentation / FreeSurfer artifacts
        # Clears lh, rh, white, pial, ribbon, parc, segment, thickness, curv, area
        surf_noise = ['surf', 'curv', 'area', 'thickness', 'white', 'pial', 'inflated', 'sphere', 'segment', 'parc', 'label', 'ribbon', 'sulcaldepth', 'vox2vox', 'xhemireg', 'leadfield']
        if any(s in k_lower for s in surf_noise) or k_lower in ['lh', 'rh']:
            continue

        # --- C. KEYWORD FILTERS ---
        if k_lower in system_noise or k_lower in task_noise or k_lower in scale_noise or k_lower in directional_noise:
            continue

        # Keep valid/recognizable BIDS suffixes (e.g., T1w, T2w, bold, dwi, magnitude, svs, ute, optodes)
        cleaned_suffix[k] = v
            
    data['suffix'] = cleaned_suffix
    return data

# Main execution logic
try:
    with open('unmapped_db_values.yaml', 'r') as f:
        cleaned_data = clean_suffixes(f.read())

    with open('schema_cleaned.yaml', 'w') as f:
        yaml.dump(cleaned_data, f, sort_keys=False, default_flow_style=False)
        
    print("Deep cleanup complete. Results saved to 'schema_cleaned.yaml'.")

except FileNotFoundError:
    print("Error: Input file not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")