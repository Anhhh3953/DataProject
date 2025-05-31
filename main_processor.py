import pandas as pd
import numpy as np
import time
import json
import os
import google.generativeai as genai
from io import StringIO
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from requests.exceptions import RequestException # Base class cho nhiều lỗi của thư viện requests (DDGS dùng requests ngầm)
from search_utils import get_random_proxy

load_dotenv()
# --Logger--
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    GOOGLE_API_KEY = os.environ.get('GOOGLR_API_KEY')
except KeyError:
    logger.error('Error with API key')
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')


# Ánh xạ tên nhà sản xuất sang tên miền chính thức
MANUFACTURER_DOMAINS = {
    "acer": "acer.com",
    "hp": "hp.com",
    "dell": "dell.com",
    "apple": "apple.com",
    "lenovo": "lenovo.com",
    "asus": "asus.com",
    "msi": "msi.com",
    "samsung": "samsung.com",
    "lg": "lg.com",
    "microsoft": "microsoft.com", 
    "razer": "razer.com",
    "huawei": "consumer.huawei.com" 
}
# Các trang web review laptop uy tín 
REPUTABLE_SITES = [
    "notebookcheck.net",
    "rtings.com",
    "laptopmag.com",
    "techradar.com",
    "theverge.com",
    "cnet.com",
    "pcworld.com",
    "expertreviews.co.uk"
]



def call_gemini_api(prompt_text, expect_json =False):
    """Function to call api of gemini 2.5 Flash"""
    try:
        if expect_json:
            response = llm_model.generate_content(prompt_text,
                                                  generation_config=genai.types.GenerationConfig(max_output_tokens=1024,
                                                                                                 temperature=0.1,                                                                             
                                                                                                 response_mime_type='application/json'))
            response.resolve()
            if response.parts:
                return response.text.strip()
            else:
                logger.warning(f'Warning: Gemini API returned no parts for JSON. Fallback or error {response.prompt_feedback}')
                return None
        
        else:
            response = llm_model.generate_content(prompt_text, generation_config=genai.types.GenerationConfig(temperature=0.2))
            response.resolve()
            if response.parts:
                return response.text.strip()
            else:
                logger.warning(f'Warning: Gemini API returned no parts for JSON. Fallback or error {response.prompt_feedback}')
                return None
    except Exception as e:
        logger.error(f'Error with alling Gemini API: {e}')
        if hasattr(e, 'response') and e.response: #Error from API server
            logger.error(f'API Response Error:{str(e)}')
        return None
        
        
def generate_question_with_llm(row_data, missing_cols):
    """Generate question for web searching missing information by LLMs"""
    
    known_spaces = {k: v for k, v in row_data.items() if pd.notna(v) and k not in missing_cols}
    missing_cols_str = ', '.join(missing_cols).replace('_', ' ')
    
    prompt =f"""
    Given the following laptop information:
    Name: {row_data.get('name', 'N/A')}
    Manufacturer: {row_data.get('manufacturer', 'N/A')}
    Other know specifications: {json.dumps(known_spaces, indent=2)}
    The following specification are missing: {missing_cols_str}.
    Generate a concise and natural language question to find these missing specification.
    The question should be suitable foor web searching and also for prompting another Ai to extract answers.
    Example: 
    If the name is "Laptop Acer Vivobook S14", manufacturer is 'acer' and missing specs are 'ram_storage' and 'weight',
    a good question should be: "What are the RAM storage and weight of the Laptop Acer Vivobook S14, manufactured by Acer?"
    Focus only on the missing specifications.
    
    Generated Question:
    
    """
    
    question = call_gemini_api(prompt)
    if question:
        logger.info(f'Generate question: {question}')
    return question

def get_manufactuer_official_domain(manufacturer_name):
    """Fetch the offical domain of official manufacturers"""
    
    if pd.isna(manufacturer_name) or not isinstance(manufacturer_name, str):
        return None
    return MANUFACTURER_DOMAINS.get(manufacturer_name.lower())

@retry(
        wait=wait_exponential(multiplier=2, min=5, max=45),
        stop=stop_after_attempt(4), # 1 for first creation and 3 others for retry
        retry=retry_if_exception_type(RequestException), # Retry if network error or DDGS error
        before_sleep=before_sleep_log(logger, logging.WARNING) # log before sleeping to retry
        
    ) 

def _ddg_text_with_retry(ddgs_instance, search_query, max_r):
        results = ddgs_instance.text(search_query, max_results=max_r) #Use ddgs_instance as parent
        if results is None:
            logger.warning(f"DDGS for {search_query} return None")
            return []
        return results

def search_web_ddg(query, manufacturer=None, max_results_per_source=3):
    """
    Perform a DuckDuckGo search with retries and delay.

    Args:
        query (str): The search query.
        max_results_per_source (int): Max number of results to retrieve.
        manufacturer: The manufacturer of the laptop

    Returns:
        list: A list of dicts with 'href' and 'body'.
    """
    snippets = []
    try:
        ddgs_instance = DDGS(proxies=get_random_proxy(), timeout=20)
    except Exception as e:
        logger.error(f"Error with create DDGS: {e}")
        return snippets
    
    
    logger.info(f'Searching web for {query}')\
            
    # 1. Find in official web first
    official_domain = get_manufactuer_official_domain(manufacturer)
    search_sources = []
    
    if official_domain:
        search_sources.append({
            'type':"Official site",
            "query": f"{query} site: {official_domain}",
            "log_msg": f'Searching official site {official_domain}',
            "max_res": max_results_per_source
        })
    
    search_sources.append({
        'type':"Reputable review site",
        "query": f"{query} {' OR '.join([f'site:{site}' for site in REPUTABLE_SITES])}",
        "log_msg": f'Searching reputable rreview site for {query}',
        "max_res": max_results_per_source * 2
    })
    search_sources.append({
        "type": "General Search",
        "query": query,
        "log_msg": f" Performing general search for '{query}'",
        "max_res": max_results_per_source
    })
    
    for source_info in search_sources:
        if len(snippets) >= max_results_per_source * 2 and source_info['type'] == 'General Search':
            break
        if len(snippets) >= max_results_per_source and source_info['type'] == "Reputable review site":
            pass
        logger.info(source_info['log_msg'])
        try:
            results = _ddg_text_with_retry(ddgs_instance, source_info['query'], source_info['max_res'])
            count = 0
            for r in results:
                if not any(r['href'] in s for s in snippets):
                    snippets.append(f"Source: {r['href']} \nSnippet: {r['body']}")
                    count += 1
            if count > 0:
                logger.info(f"Found {count} new results form {source_info['type']}")
        except Exception as e:
            logger.error(f"Failed to search {source_info['type']} afterr retries: {e}")
            time.sleep(1) # Small delay between different types of searches                        
        
    return snippets[:max_results_per_source*3] #limit the return snippets

    
    
    
    
def extract_answers_with_llm(question, search_snippets, missing_cols):
    """Extract answers from snippets with LLMs, return json"""
    
    if not search_snippets:
        return None
    
    snippets_str = "\n\n".join([f'---Snippet {i+1}---\n{s}' for i, s in enumerate(search_snippets)])
    missing_cols_json_keys = [col.replace(' ', "_") for col in missing_cols]
    prompt = f"""
    You are an AI assistant tasked with extracting specific information about laptop based on provided search snippets.
    The user asked this question: "{question}"
    The specific information to extract is for these fields: {', '.join(missing_cols)}.
    Here are the search result snippets:
    {snippets_str}
    
    Base *strictly and only* on the provided search result snippets, extract the values for the mising fields.
    - If a value for a specific field is found, provide it
    - If a value for a specific field cannot be definitvely found in the snippets, use 'null' for that field's value.
    - Ensure the outcome is a valid JSON object
    - The keys in the JSON object must be exactly: {json.dumps(missing_cols_json_keys)}.
    - Do not infer or make up information not present in the snippets.
    Example for missing fields "ram_storage" and "weight":
    {{
      "ram_storage": "16GB DDR4",
      "weight": "1.5 kg"
    }}
    If weight is not found:
    {{
      "ram_storage": "16GB DDR4",
      "weight": null
    }}
    
    """
    json_output_str = call_gemini_api(prompt, expect_json=True)
    if json_output_str:
        try:
            # if is markdown
            if json_output_str.startswith("```json```"):
                json_output_str = json_output_str.strip("```json").strip("`").strip()
            extracted_data = json.loads(json_output_str)
            # Đảm bảo tất cả các key mong muốn đều có, nếu không thì thêm với giá trị None
            # (Dù prompt đã yêu cầu nhưng LLM có thể không tuân theo hoàn toàn)
            final_data = {key: extracted_data.get(key) for key in missing_cols_json_keys}
            return final_data
        except json.JSONDecodeError as e:
            logger.error(f'Error to decode JSON from LLMs: {e}')
            logger.error(f'LLMs raw output: {json_output_str}')
            return None
    return None



def process_laptop_data(df, potential_fill_columns):
    """Process DataFrame and fill in missing data"""
    
    df_updated = df.copy()
    for col in potential_fill_columns:
        if col not in df_updated.columns:
            df_updated[col] = pd.NA
    
    rows_to_process_indices = df_updated[df_updated[potential_fill_columns].isnull().any(axis=1)].index
    logger.info(f"Found {len(rows_to_process_indices)} rows with missing data in target columns.")
    
    # Batch handling (rows in DataFrame not batch API for LLMs)
    batch_size = 5
    num_batches = (len(rows_to_process_indices) + batch_size -1) // batch_size
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(rows_to_process_indices))
        current_batch_indices = rows_to_process_indices[start_idx:end_idx] 
        logger.info(f'Processing batch {batch_num + 1}/{num_batches}. Rows {current_batch_indices.tolist()}')
        
        for index in current_batch_indices:
            row = df_updated.loc[index]
            logger.info(f"Processing row index {index}: {row.get('name', 'NA')}")
            
            # Fetch missingvalue columns in potential_fill_columns for current row
            current_missing_cols = [col for col in potential_fill_columns if pd.isnull(row[col])]
            if not current_missing_cols:
                logger.info(f'Skip row {index}, now missing target data')
                continue
            logger.info(f'Missing columns: {current_missing_cols}')
            # 1. Generate question by LLM
            question = generate_question_with_llm(row.to_dict(), current_missing_cols)
            if not question:
                logger.warning(f'Cannot generate question for index {index} -> Skip')
                time.sleep(2)
                continue
            
            # 2. Search web
            # DDGS() instance được tạo mới mỗi lần trong search_web_ddg
            search_snippets = search_web_ddg(question, row.get('manufacturer'))
            if not search_snippets: 
                logger.warning(f'Could not find web searchign results for question: {question} --> Skip')
                time.sleep(1)
                continue
            logger.info(f'Found {len(search_snippets)} for web searching')
            
            # 3. Extract answer
            extracted_info = extract_answers_with_llm(question, search_snippets, current_missing_cols)
            if extracted_info:
                logger.info(f'Extracted info: {extracted_info}')
                for col, value in extracted_info.items():
                    if value is not None:
                        df_updated.loc[index, col] = value # only update if LLMs could find answers
                    
                    else:
                        logger.warning(f'Cannot extract info for row {index} from snippets. ')
                    time.sleep(5)
                        
        logger.info(f'Finish batch {batch_num + 1}/{num_batches}')
    return df_updated
        
        
        
    
    
    
    
    # site_query = f'{query} site: {official_domain}'
    #     logger.info(f'Searching official site: {site_query}')
    #     try:
    #         results = _ddg_text_with_retry.text(site_query, max_results=max_results_per_source)
    #         for r in results:
    #             snippets.append(f"Source:{r['href']} \nSnippets: {r['body']}")
    #         if snippets: 
    #             logger.info(f'Found {len(results)} results from official site')
        
    #     except Exception as e:
    #         logger.error(f'Error searching official site {official_domain}: {e}')
    #         time.sleep(1)
            
    # # 2. If not enough, find further information in reputable review sites
    # if len(snippets) < max_results_per_source * 1: #increase if needed
    #     site_filters = ' OR '.join([f'site:{site}' for site in REPUTABLE_SITES])
    #     reputable_query = f'{query} ({site_filters})'
    #     logger.info(f"Searching reputable sites for: '{query}'...")

    #     try:
    #         results = _ddg_text_with_retry.text(reputable_query, max_results=max_results_per_source*2)
    #         for r in results:
    #             if not any (r['href'] in s for s in snippets):
    #                 snippets.append(f"Source: {r['href']} \nSnippets: {r['body']}")
    #         if results: 
    #             logger.info(f"Found {len(results)} relevant results from reputable sites.")
    #     except Exception as e:
    #         logger.error(f'Error searching reputable sites: {e}')
    #         time.sleep(3)
    
    # # 3. If still not enough, just fk search
    # if len(results) < max_results_per_source *2:
    #     logger.info(f'Performing general search for: {query}')
    #     try:
    #         results = _ddg_text_with_retry.text(query, max_results=max_results_per_source)
    #         for r in results:
    #             if not any(r['href'] in s for s in snippets):
    #                 snippets.append(f"Source: {r['href']}\nSnippet: {r['body']}")
    #         if results: logger.info(f" Found {len(results)} results from general search.")
            
    #     except Exception as e:
    #          logger.error(f"Error during general search: {e}")
    #     time.sleep(1)