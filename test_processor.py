import pandas as pd
from io import StringIO
from main_processor import process_laptop_data
from dotenv import load_dotenv
import logging
import os

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("filling_null.log", mode='w', encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_configuration():
    load_dotenv()
    config = {}
    config['input_file'] = os.getenv('INPUT_FILE_PATH')
    config['output_file'] = os.getenv('OUTPUT_FILE_PATH')
    config['google_api_key'] = os.getenv('GOOGLE_API_KEY')
    potential_fill_columns_str = os.getenv('POTENTIAL_FILL_COLUMNS')
    if not all([config['input_file'], config['output_file'], config['google_api_key'], potential_fill_columns_str]):
        logger.error("ERROR: Something is wrrong with the environment variable")
        return None
    config['potential_fill_columns'] = [col.strip() for col in potential_fill_columns_str.split(',') if col.strip()]
    logger.info("Configuration success")
    logger.info(f"  Input file: {config['input_file']}")
    logger.info(f"  Output file: {config['output_file']}")
    logger.info(f"  Potential columns to fill: {config['potential_fill_columns']}") 

    return config

def load_data_from_file(filepath):
    try:
        df = pd.read_csv(filepath)
        df = df.replace('', pd.NA).fillna(pd.NA)
        logger.info(f'Loaded data from {filepath}. Columns: {len(df.columns)}. Rows: {len(df)}')
        return df
    except FileNotFoundError:
        logger.error(f'Could not find input file')
    except pd.errors.EmptyDataError:
        logger.error(f"File {filepath} is empty.")
        return None
    except Exception as e:
        logger.error(f"Error with downloading file {filepath}: {e}")
        return None
    
def ensure_output_dir(filepath):
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedir(output_dir)
            logger.info(f'Create output dir: {output_dir}')
        except OSError as e:
            logger.error(f"Error creating output dir {output_dir}: {e}")
            return False
    return True

def main():
    config = load_configuration()
    if not config: return
    
    df_laptop_origin = load_data_from_file(config['input_file'])
    if df_laptop_origin is None: 
        return
    
    df_filled = process_laptop_data(df_laptop_origin, config['potential_fill_columns'])
    if ensure_output_dir(config['output_file']):
        try: 
            df_filled.to_csv(config['output_file'], index=False)
            logger.info(f"Sucessfully saved to {config['output_file']}")
        except Exception as e:
            logger.error(f"Failed to save {config['output_file']}")
    else:
        logger.error('Cannot create output dir')
    logger.info('Process completed')
    
    
if __name__ == '__main__':
    main()